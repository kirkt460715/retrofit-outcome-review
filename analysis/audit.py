"""观测层面的水平移动审计。

观测序列带有稳定的周循环。对每个住户，先用最早一段无设施周拟合循环相位，
再把每周的观察值减去该相位应有的值，得到残差。残差连续两处以上非零，
说明该住户的出行机会出现了登记没有解释的水平移动：

- 移动出现在某个阶段开放周，且登记组合确实新增了设施：属于可用性变化，正常；
- 移动出现在没有组合变化的周：说明存在未登记的实际可用范围（例如共用通道），
  该楼栋楼层不能继续当作未受影响的对照；
- 登记关闭后残差仍不回落：说明“关闭”只是登记状态，实际仍可用，间断周
  不能当作无暴露基线，也不能按 0 次出门处理。

这里只做筛查与口径标记，不据此给出因果结论。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .data_access import Tables
from .estimate import Cycle, observation_lookup, person_lookup
from .model import Availability

MIN_RUN = 2


@dataclass(frozen=True)
class LevelShift:
    person: str
    building: str
    floor: int
    onset: str
    end: str
    size: int
    phases_at_onset: tuple[str, ...]
    combo_at_onset: str
    combo_before: str
    explanation: str

    @property
    def explained(self) -> bool:
        return self.explanation == "开放"

    @property
    def cell(self) -> tuple[str, int]:
        return (self.building, self.floor)


def fit_phases(
    tables: Tables, cycle: Cycle, fit_weeks: Sequence[str]
) -> dict[str, int]:
    """用拟合周的多数票确定每个住户的循环相位。"""
    observations = observation_lookup(tables)
    weeks = sorted({row["week"] for row in tables.observations})
    index = {week: i for i, week in enumerate(weeks)}
    phases: dict[str, int] = {}
    for person in person_lookup(tables):
        votes = [0] * cycle.length
        for week in fit_weeks:
            record = observations.get((person, week))
            if not record or record["state"] != "observed":
                continue
            value = int(record["outings"])
            for phase in range(cycle.length):
                if value == (index[week] + phase) % cycle.length:
                    votes[phase] += 1
        if any(votes):
            phases[person] = max(range(cycle.length), key=lambda p: votes[p])
    return phases


def fit_weeks(availability: Availability, count: int = 4) -> list[str]:
    """最早一段所有单元都无设施、无施工、无其它项目的周。"""
    out: list[str] = []
    for week in availability.weeks:
        if all(
            availability.baseline_eligible(building, floor, week)
            for building in availability.floors
            for floor in availability.floors[building]
        ):
            out.append(week)
        if len(out) >= count:
            break
    if out:
        return out
    # 退回最早 count 周，至少保证相位可拟合。
    return list(availability.weeks[:count])


def detect_shifts(
    availability: Availability, cycle: Cycle
) -> list[LevelShift]:
    if not cycle.usable:
        return []
    tables = availability.tables
    observations = observation_lookup(tables)
    households = person_lookup(tables)
    weeks = availability.weeks
    index = {week: i for i, week in enumerate(weeks)}
    phases = fit_phases(tables, cycle, fit_weeks(availability))

    shifts: list[LevelShift] = []
    for person, row in households.items():
        phase = phases.get(person)
        if phase is None:
            continue
        residuals: list[tuple[int, int]] = []
        for week in weeks:
            record = observations.get((person, week))
            if not record or record["state"] != "observed":
                continue
            expected = (index[week] + phase) % cycle.length
            delta = int(record["outings"]) - expected
            if delta:
                residuals.append((index[week], delta))

        runs: list[list[tuple[int, int]]] = []
        run: list[tuple[int, int]] = []
        for item in residuals:
            if run and item[0] == run[-1][0] + 1:
                run.append(item)
            else:
                if run:
                    runs.append(run)
                run = [item]
        if run:
            runs.append(run)

        merged: list[list[tuple[int, int]]] = []
        for segment in runs:
            if merged and _can_bridge(availability, person, row, merged[-1], segment, weeks):
                merged[-1] = merged[-1] + segment
            else:
                merged.append(segment)
        for segment in merged:
            if len(segment) >= MIN_RUN:
                shifts.append(
                    _shift(availability, person, row, segment, weeks, observations)
                )
    return sorted(shifts, key=lambda s: (s.building, s.floor, s.onset, s.person))


def _can_bridge(
    availability: Availability,
    person: str,
    row: Mapping[str, str],
    earlier: Sequence[tuple[int, int]],
    later: Sequence[tuple[int, int]],
    weeks: Sequence[str],
) -> bool:
    """缺报只隔一两周、方向相同、且组合没有减少时，把两段视为同一次水平移动。

    如果后来的一段比先前少了设施，则不能合并：那正是“关闭后未回落”的证据。
    """
    gap = later[0][0] - earlier[-1][0] - 1
    if gap < 0 or gap > 2:
        return False
    if later[0][1] != earlier[-1][1]:
        return False
    building, floor = row["building"], int(row["floor"])
    before = availability.combo(building, floor, weeks[earlier[0][0]])
    after = availability.combo(building, floor, weeks[later[0][0]])
    return before <= after


def _last_observed_before(
    observations: Mapping[tuple[str, str], Mapping[str, str]],
    person: str,
    weeks: Sequence[str],
    start: int,
) -> str:
    for i in range(start - 1, -1, -1):
        record = observations.get((person, weeks[i]))
        if record and record["state"] == "observed":
            return weeks[i]
    return weeks[start]


def _shift(
    availability: Availability,
    person: str,
    row: Mapping[str, str],
    run: Sequence[tuple[int, int]],
    weeks: Sequence[str],
    observations: Mapping[tuple[str, str], Mapping[str, str]],
) -> LevelShift:
    start = run[0][0]
    end = run[-1][0]
    onset = weeks[start]
    building, floor = row["building"], int(row["floor"])
    exposure = availability.exposure(building, floor, onset)
    prior_week = _last_observed_before(observations, person, weeks, start)
    prior = availability.exposure(building, floor, prior_week)
    added = exposure.kinds - prior.kinds
    removed = prior.kinds - exposure.kinds
    size = int(round(sum(delta for _, delta in run) / len(run)))

    # 判断只看“增/撤了什么”和“移动方向”，不看登记状态本身：
    # 登记关闭不等于实际停用，施工也不一定解释上行。
    if added:
        explanation = "开放"
    elif removed and size > 0:
        explanation = "关闭后未回落"
    elif removed and size < 0:
        explanation = "撤除"
    elif exposure.under_construction and size < 0:
        explanation = "施工封闭"
    elif size > 0:
        explanation = "未登记的实际可用变化"
    else:
        explanation = "未登记的实际可用变化"

    return LevelShift(
        person=person,
        building=building,
        floor=floor,
        onset=onset,
        end=weeks[end],
        size=size,
        phases_at_onset=exposure.phase_ids,
        combo_at_onset=exposure.combo,
        combo_before=prior.combo,
        explanation=explanation,
    )


@dataclass
class Contamination:
    """按楼栋-楼层的登记未解释水平移动。"""

    intervals: dict[tuple[str, int], list[LevelShift]]

    @classmethod
    def from_shifts(cls, shifts: Iterable[LevelShift]) -> "Contamination":
        intervals: dict[tuple[str, int], list[LevelShift]] = {}
        for shift in shifts:
            if shift.explained:
                continue
            intervals.setdefault(shift.cell, []).append(shift)
        return cls(intervals)

    def at(self, building: str, floor: int, week: str) -> LevelShift | None:
        for shift in self.intervals.get((building, floor), []):
            if shift.onset <= week <= shift.end:
                return shift
        return None

    def cells_overlapping(
        self, weeks: Iterable[str]
    ) -> dict[tuple[str, int], list[LevelShift]]:
        wanted = list(weeks)
        out: dict[tuple[str, int], list[LevelShift]] = {}
        for cell, shifts in self.intervals.items():
            hits = [
                shift
                for shift in shifts
                if any(shift.onset <= w <= shift.end for w in wanted)
            ]
            if hits:
                out[cell] = hits
        return out


def contamination_summary(
    availability: Availability, shifts: Sequence[LevelShift]
) -> list[dict]:
    """把同一楼栋-楼层、同一起点的移动合并成一行，便于登记与回查。"""
    grouped: dict[tuple[str, int, str, str, str], list[LevelShift]] = {}
    for shift in shifts:
        key = (
            shift.building,
            shift.floor,
            shift.onset,
            shift.end,
            shift.explanation,
        )
        grouped.setdefault(key, []).append(shift)
    rows = []
    for (building, floor, onset, end, explanation), items in sorted(grouped.items()):
        rows.append(
            {
                "building": building,
                "floor": floor,
                "onset": onset,
                "end": end,
                "size": items[0].size,
                "people": len(items),
                "explanation": explanation,
                "phases_at_onset": items[0].phases_at_onset,
                "combo_before": items[0].combo_before,
                "combo_at_onset": items[0].combo_at_onset,
            }
        )
    return rows


def observed_close_mismatches(
    availability: Availability, shifts: Sequence[LevelShift]
) -> list[dict]:
    """登记已关闭、但该楼栋楼层仍有正向水平移动的周，说明关闭只是登记状态。"""
    out = []
    for phase in availability.tables.phases:
        if not phase["close"]:
            continue
        close = phase["close"]
        cells = availability.served_cells(
            phase["way"],
            int(phase["first_floor"]),
            int(phase["last_floor"]),
            phase["actual"],
        )
        for cell in cells:
            ongoing = [
                shift
                for shift in shifts
                if shift.cell == cell
                and shift.size > 0
                and shift.onset < close <= shift.end
            ]
            if ongoing:
                out.append(
                    {
                        "phase": phase["id"],
                        "cell": f"{cell[0]} f{cell[1]}",
                        "close": close,
                        "shift_onset": ongoing[0].onset,
                        "shift_end": ongoing[0].end,
                        "size": ongoing[0].size,
                        "people": len(ongoing),
                        "issue": "登记关闭后观察水平仍未回落，间断周不能当作无暴露基线",
                    }
                )
    return out
