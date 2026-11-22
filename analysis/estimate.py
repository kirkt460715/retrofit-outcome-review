"""阶段效果估计：周期配对、对照重核、住户与楼栋两级不确定性。

观测可能带有稳定的周循环（例如按周轮换的活动安排）。若直接用前后均值
比较，循环相位错位会制造伪效应。因此主估计先按周期位置配对，再做差中差；
前后均值差只作为敏感性展示。

估计单位是住户；楼栋是重抽、置换与对照筛查的聚类单位。
缺报与搬离不进入主估计，只通过填补边界展示其影响。
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .data_access import Tables
from .model import Availability, Opening

MAX_OUTINGS = 5


# ---------------------------------------------------------------------------
# 周期识别
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cycle:
    length: int
    correlation: float
    method: str

    @property
    def usable(self) -> bool:
        return self.length > 1 and self.correlation >= 0.6


def detect_cycle(tables: Tables, max_lag: int = 10) -> Cycle:
    """用住户内去均值的自相关识别周循环。

    对每个滞后阶数，汇总所有住户的配对相关；取相关最高且超过 0.6 的阶数。
    识别不到时退回长度 1（不做配对），并在报告中说明。
    """
    weeks = sorted({row["week"] for row in tables.observations})
    position = {week: i for i, week in enumerate(weeks)}
    series: dict[str, dict[int, int]] = {}
    for row in tables.observations:
        if row["state"] != "observed":
            continue
        series.setdefault(row["person"], {})[position[row["week"]]] = int(row["outings"])
    if len(weeks) < 6:
        return Cycle(1, 0.0, "周数不足")

    best_lag = 1
    best_r = 0.0
    for lag in range(2, min(max_lag, len(weeks) - 3) + 1):
        correlations = []
        for values in series.values():
            if len(values) < 6:
                continue
            mean = statistics.mean(values.values())
            pairs = [
                (values[i] - mean, values[i + lag] - mean)
                for i in values
                if i + lag in values
            ]
            if len(pairs) < 5:
                continue
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            sx = sum(x * x for x in xs) ** 0.5
            sy = sum(y * y for y in ys) ** 0.5
            if sx and sy:
                correlations.append(sum(x * y for x, y in pairs) / (sx * sy))
        if not correlations:
            continue
        r = statistics.mean(correlations)
        if r > best_r:
            best_r = r
            best_lag = lag
    if best_r >= 0.6:
        return Cycle(best_lag, best_r, "住户内自相关")
    return Cycle(1, best_r, "未识别到稳定循环")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonChange:
    person: str
    building: str
    floor: int
    group: str
    change: float | None
    pre_mean: float | None
    post_mean: float | None
    paired_residues: int
    pre_observed: int
    post_observed: int
    pre_missing: int
    post_missing: int
    moved: bool
    unmatched_change: float | None = None

    @property
    def usable(self) -> bool:
        return self.change is not None


@dataclass
class Estimate:
    label: str
    reference: str
    target: str
    floors: list[int]
    pre_weeks: list[str]
    post_weeks: list[str]
    treated_cells: list[tuple[str, int]]
    treated_buildings: list[str]
    control_buildings: list[str]
    control_audit: list[dict]
    treated: list[PersonChange]
    control: list[PersonChange]
    treated_change: float | None
    control_change: float | None
    did: float | None
    unmatched_did: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    building_did: float | None
    building_ci_low: float | None
    building_ci_high: float | None
    pre_trend_treated: float | None
    pre_trend_control: float | None
    pre_trend_diff: float | None
    missing_bound_low: float | None
    missing_bound_high: float | None
    treated_moved: int
    control_moved: int
    treated_missing: int
    control_missing: int
    treated_observed_rate: float
    control_observed_rate: float
    cycle: Cycle
    flagged_controls: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confounded: bool = False
    recovery: bool = False
    sensitivity: bool = False

    @property
    def control_building_count(self) -> int:
        return len({c.building for c in self.control if c.usable})

    @property
    def n_treated(self) -> int:
        return sum(1 for c in self.treated if c.usable)

    @property
    def n_control(self) -> int:
        return sum(1 for c in self.control if c.usable)

    @property
    def sufficient(self) -> bool:
        return (
            len(self.pre_weeks) >= 2
            and len(self.post_weeks) >= 2
            and self.n_treated >= 2
            and self.n_control >= 2
            and self.control_building_count >= 2
            and self.paired_ok
        )

    @property
    def paired_ok(self) -> bool:
        """周期配对是否足以解读主估计。

        每个可用住户至少要有一个同循环位置的前后配对；若超过两成住户
        只有一个配对位置，视为配对偏弱，但仍在报告中给出数字并标注。
        """
        if not self.cycle.usable:
            return True
        usable = [c for c in self.treated + self.control if c.usable]
        if not usable:
            return False
        if any(c.paired_residues < 1 for c in usable):
            return False
        thin = sum(1 for c in usable if c.paired_residues < 2)
        return thin / len(usable) <= 0.2


    @property
    def evidence(self) -> str:
        if self.quarantined:
            return "混杂：处理单元存在登记未解释的可用变化"
        if self.confounded:
            return "混杂，不能单独归因"
        if self.recovery:
            return "恢复开放，不能当作首次效果"
        if not self.sufficient or self.did is None:
            return "无足够可比观察"
        if self.ci_low is None or self.ci_high is None:
            return "仅点估计"
        if self.ci_low > 0:
            return "正向且区间不含 0"
        if self.ci_high < 0:
            return "负向且区间不含 0"
        return "区间含 0"


# ---------------------------------------------------------------------------
# 取值与周期配对
# ---------------------------------------------------------------------------


def person_lookup(tables: Tables) -> dict[str, Mapping[str, str]]:
    return {row["person"]: row for row in tables.households}


def observation_lookup(tables: Tables) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(row["person"], row["week"]): row for row in tables.observations}


def person_change(
    person: str,
    row: Mapping[str, str],
    pre: Sequence[str],
    post: Sequence[str],
    observations: Mapping[tuple[str, str], Mapping[str, str]],
    cycle: Cycle,
    week_index: Mapping[str, int],
    pre_fill: float | None = None,
    post_fill: float | None = None,
) -> PersonChange:
    """周期配对变化；pre_fill/post_fill 用于缺报边界（0 或观察最大值）。

    搬离者不填补，change 设为 None 并从该期移出。
    """
    def residue(week: str) -> int:
        return week_index[week] % cycle.length if cycle.usable else 0

    pre_values: dict[int, list[float]] = {}
    post_values: dict[int, list[float]] = {}
    pre_observed = post_observed = 0
    pre_missing = post_missing = 0
    moved = False

    for week in pre:
        record = observations.get((person, week))
        if record is None or record["state"] == "missing":
            pre_missing += 1
            if pre_fill is not None:
                pre_values.setdefault(residue(week), []).append(pre_fill)
        elif record["state"] == "moved":
            moved = True
        else:
            pre_observed += 1
            pre_values.setdefault(residue(week), []).append(int(record["outings"]))

    for week in post:
        record = observations.get((person, week))
        if record is None or record["state"] == "missing":
            post_missing += 1
            if post_fill is not None:
                post_values.setdefault(residue(week), []).append(post_fill)
        elif record["state"] == "moved":
            moved = True
        else:
            post_observed += 1
            post_values.setdefault(residue(week), []).append(int(record["outings"]))

    if moved or not pre_observed or not post_observed:
        change = None
    else:
        diffs = [
            statistics.mean(post_values[r]) - statistics.mean(pre_values[r])
            for r in sorted(set(pre_values) & set(post_values))
        ]
        change = statistics.mean(diffs) if diffs else None

    paired = len(set(pre_values) & set(post_values))

    def mean_of(values: Mapping[int, list[float]]) -> float | None:
        flat = [v for rows in values.values() for v in rows]
        return statistics.mean(flat) if flat else None

    pre_mean = mean_of(pre_values)
    post_mean = mean_of(post_values)
    unmatched = (
        post_mean - pre_mean
        if pre_mean is not None and post_mean is not None and not moved
        else None
    )

    return PersonChange(
        person=person,
        building=row["building"],
        floor=int(row["floor"]),
        group=row["group"],
        change=change,
        pre_mean=pre_mean,
        post_mean=post_mean,
        paired_residues=paired,
        pre_observed=pre_observed,
        post_observed=post_observed,
        pre_missing=pre_missing,
        post_missing=post_missing,
        moved=moved,
        unmatched_change=unmatched,
    )


def people_in_cells(tables: Tables, cells: Iterable[tuple[str, int]]) -> list[str]:
    wanted = set(cells)
    return [
        row["person"]
        for row in tables.households
        if (row["building"], int(row["floor"])) in wanted
    ]


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------


def pre_window(
    availability: Availability,
    cell: tuple[str, int],
    reference: frozenset[str],
    stop_before: str,
    max_weeks: int = 8,
) -> list[str]:
    """从 stop_before 的前一周向前，取组合恒为 reference 的连续干净周。"""
    building, floor = cell
    ordered = availability.weeks
    if stop_before not in ordered:
        stop_before = next((w for w in ordered if w >= stop_before), ordered[-1])
    i = ordered.index(stop_before) - 1
    out: list[str] = []
    while i >= 0 and len(out) < max_weeks:
        week = ordered[i]
        if not availability.clean_week(building, floor, week, reference):
            break
        out.append(week)
        i -= 1
    return list(reversed(out))


def post_window(
    availability: Availability,
    cell: tuple[str, int],
    target: frozenset[str],
    anchor: str,
    max_weeks: int = 4,
    include_other_projects: bool = False,
) -> list[str]:
    """从 anchor 起，取目标组合持续、无施工/间断的连续周。"""
    building, floor = cell
    ordered = availability.weeks
    if anchor not in ordered:
        anchor = next((w for w in ordered if w >= anchor), None)
        if anchor is None:
            return []
    i = ordered.index(anchor)
    out: list[str] = []
    while i < len(ordered) and len(out) < max_weeks:
        week = ordered[i]
        exposure = availability.exposure(building, floor, week)
        ok = (
            exposure.kinds == target
            and not exposure.under_construction
            and not exposure.interrupted
        )
        if ok and not include_other_projects and availability.other_project_overlap(building, week):
            ok = False
        if not ok:
            break
        out.append(week)
        i += 1
    return out


# ---------------------------------------------------------------------------
# 对照审计与污染筛查
# ---------------------------------------------------------------------------


def control_audit(
    availability: Availability,
    treated_buildings: Iterable[str],
    floors: Iterable[int],
    pre: Sequence[str],
    post: Sequence[str],
    contaminated: Mapping[str, str] | None = None,
    contamination=None,
) -> list[dict]:
    """逐楼栋说明能否作为对照，并给出第一处失败原因。

    contamination 是 audit.Contamination，用来识别“登记没有解释的可用变化”：
    这类楼栋楼层即使登记组合仍是空，也不能当未受影响的对照。
    """
    treated = set(treated_buildings)
    floors = sorted(set(floors))
    contaminated = contaminated or {}
    audit = []
    window = list(pre) + list(post)
    for building in sorted(availability.floors):
        row = {
            "building": building,
            "way": availability.building_index[building]["way"],
            "area": availability.building_index[building]["area"],
            "baseline_group": availability.building_index[building]["baseline_group"],
            "status": "candidate",
            "reason": "",
        }
        if building in treated:
            row["status"] = "treated"
            row["reason"] = "在同一开放事件的服务范围内"
            audit.append(row)
            continue
        if building in contaminated:
            row["status"] = "excluded"
            row["reason"] = contaminated[building]
            audit.append(row)
            continue
        if not set(floors) <= set(availability.floors[building]):
            row["status"] = "excluded"
            row["reason"] = "楼层不匹配"
            audit.append(row)
            continue
        if contamination is not None:
            hits = contamination.cells_overlapping(window)
            hit = next((cell for cell in hits if cell[0] == building and cell[1] in floors), None)
            if hit is not None:
                shift = hits[hit][0]
                row["status"] = "excluded"
                row["reason"] = (
                    f"{shift.onset} 起出现登记未解释的可用变化：{shift.explanation}"
                )
                audit.append(row)
                continue
        reason = ""
        for week in window:
            hits = availability.other_project_overlap(building, week)
            if hits:
                reason = f"{week} 其他项目 {'、'.join(hits)}"
                break
            for floor in floors:
                exposure = availability.exposure(building, floor, week)
                if exposure.under_construction:
                    reason = f"{week} 施工 {'、'.join(exposure.construction_phases)}"
                    break
                if exposure.interrupted:
                    reason = f"{week} 开放后间断"
                    break
                if exposure.kinds:
                    reason = f"{week} 已有设施 {'+'.join(sorted(exposure.kinds))}"
                    break
            if reason:
                break
        if reason:
            row["status"] = "excluded"
            row["reason"] = reason
        audit.append(row)
    return audit


def building_changes(changes: Iterable[PersonChange]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for change in changes:
        if change.usable:
            grouped.setdefault(change.building, []).append(change.change)
    return {b: statistics.mean(v) for b, v in grouped.items() if v}


def screen_synchronized_beneficiaries(
    control_buildings: Sequence[str],
    building_value: Mapping[str, float],
) -> list[dict]:
    """对照候选里与开放周同步的大幅变化，视为疑似未登记共用通道。

    这是筛查而非因果识别：只用于把该楼栋移出对照并提示登记缺口。
    """
    values = [building_value[b] for b in control_buildings if b in building_value]
    if len(values) < 3:
        return []
    mean = statistics.mean(values)
    sd = statistics.pstdev(values) or 0.0
    cutoff = mean + max(3 * sd, 0.5)
    flagged = []
    for building in control_buildings:
        value = building_value.get(building)
        if value is not None and value >= cutoff:
            flagged.append(
                {
                    "building": building,
                    "change": round(value, 3),
                    "cutoff": round(cutoff, 3),
                    "issue": "对照候选与开放周同步大幅变化，疑似经未登记共用通道受益",
                }
            )
    return flagged


# ---------------------------------------------------------------------------
# 不确定性
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def cluster_bootstrap(
    treated: Sequence[float], control: Sequence[float], seed: int, reps: int = 2000
) -> tuple[float, float]:
    rng = random.Random(seed)
    if len(treated) < 2 or len(control) < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(reps):
        t = [rng.choice(treated) for _ in treated]
        c = [rng.choice(control) for _ in control]
        draws.append(statistics.mean(t) - statistics.mean(c))
    return (_percentile(draws, 0.025), _percentile(draws, 0.975))


def permutation_p(
    treated: Sequence[float], control: Sequence[float], seed: int, reps: int = 5000
) -> float | None:
    if len(treated) < 2 or len(control) < 2:
        return None
    pooled = list(treated) + list(control)
    n = len(treated)
    observed = abs(statistics.mean(treated) - statistics.mean(control))
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pooled)
        diff = abs(statistics.mean(pooled[:n]) - statistics.mean(pooled[n:]))
        if diff >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (reps + 1)


def within_person_slope(
    people: Iterable[str],
    weeks: Sequence[str],
    observations: Mapping[tuple[str, str], Mapping[str, str]],
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for person in people:
        pairs = []
        for position, week in enumerate(weeks):
            record = observations.get((person, week))
            if record and record["state"] == "observed":
                pairs.append((position, int(record["outings"])))
        if len(pairs) < 2:
            continue
        mean_x = statistics.mean(x for x, _ in pairs)
        mean_y = statistics.mean(y for _, y in pairs)
        for x, y in pairs:
            numerator += (x - mean_x) * (y - mean_y)
            denominator += (x - mean_x) ** 2
    if denominator == 0:
        return None
    return numerator / denominator


def missing_bounds(
    person: str,
    row: Mapping[str, str],
    pre: Sequence[str],
    post: Sequence[str],
    observations: Mapping[tuple[str, str], Mapping[str, str]],
    cycle: Cycle,
    week_index: Mapping[str, int],
) -> tuple[float | None, float | None]:
    """同一周期配对下，把缺报周填到最不利/最有利两端，得到个人变化的范围。

    下界：干预前缺报按观察最大值填、干预后缺报按 0 填；
    上界：干预前缺报按 0 填、干预后缺报按观察最大值填。
    搬离者不填补，change 为 None。
    """
    low = person_change(
        person, row, pre, post, observations, cycle, week_index,
        pre_fill=float(MAX_OUTINGS), post_fill=0.0,
    )
    high = person_change(
        person, row, pre, post, observations, cycle, week_index,
        pre_fill=0.0, post_fill=float(MAX_OUTINGS),
    )
    if low.change is None or high.change is None:
        return (None, None)
    return (low.change, high.change)


# ---------------------------------------------------------------------------
# 核心估计
# ---------------------------------------------------------------------------


def estimate_core(
    availability: Availability,
    label: str,
    treated_cells: Sequence[tuple[str, int]],
    reference: frozenset[str],
    target: frozenset[str],
    pre: Sequence[str],
    post: Sequence[str],
    cycle: Cycle,
    seed: int = 20260105,
    contaminated: Mapping[str, str] | None = None,
    contamination=None,
    notes: Sequence[str] = (),
    confounded: bool = False,
    recovery: bool = False,
    sensitivity: bool = False,
    group: str | None = None,
) -> Estimate:
    tables = availability.tables
    households = person_lookup(tables)
    observations = observation_lookup(tables)
    week_index = {week: i for i, week in enumerate(availability.weeks)}
    floors = sorted({floor for _, floor in treated_cells})
    treated_buildings = sorted({building for building, _ in treated_cells})

    audit = control_audit(
        availability, treated_buildings, floors, pre, post, contaminated, contamination
    )
    candidates = [row["building"] for row in audit if row["status"] == "candidate"]

    # 处理单元自身若有登记未解释的水平移动，这个组合的估计被混杂；
    # 但仍保留数字，便于会议现场解释为什么不能用。
    quarantined = []
    if contamination is not None:
        hits = contamination.cells_overlapping(list(pre) + list(post))
        for cell in treated_cells:
            for shift in hits.get(cell, []):
                if shift.explanation != "开放":
                    quarantined.append(
                        {
                            "cell": f"{cell[0]} f{cell[1]}",
                            "onset": shift.onset,
                            "end": shift.end,
                            "explanation": shift.explanation,
                        }
                    )
    if quarantined:
        confounded = True

    treated_people = people_in_cells(tables, treated_cells)
    if group is not None:
        treated_people = [p for p in treated_people if households[p]["group"] == group]
    treated = [
        person_change(p, households[p], pre, post, observations, cycle, week_index)
        for p in treated_people
    ]
    candidate_control = [
        person_change(p, households[p], pre, post, observations, cycle, week_index)
        for building in candidates
        for p in households
        if households[p]["building"] == building and int(households[p]["floor"]) in floors
    ]
    if group is not None:
        candidate_control = [
            c for c in candidate_control if households[c.person]["group"] == group
        ]
    candidate_values = building_changes(candidate_control)
    flagged = screen_synchronized_beneficiaries(
        candidates, {**candidate_values, **building_changes(treated)}
    )
    flagged_buildings = {row["building"] for row in flagged}
    control_buildings = [b for b in candidates if b not in flagged_buildings]
    control = [c for c in candidate_control if c.building in set(control_buildings)]

    t_values = [c.change for c in treated if c.usable]
    c_values = [c.change for c in control if c.usable]
    t_mean = statistics.mean(t_values) if t_values else None
    c_mean = statistics.mean(c_values) if c_values else None
    did = t_mean - c_mean if t_mean is not None and c_mean is not None else None

    unmatched_t = [c.unmatched_change for c in treated if c.unmatched_change is not None]
    unmatched_c = [c.unmatched_change for c in control if c.unmatched_change is not None]
    unmatched_did = (
        statistics.mean(unmatched_t) - statistics.mean(unmatched_c)
        if unmatched_t and unmatched_c
        else None
    )

    t_buildings = list(building_changes(treated).values())
    c_buildings = list(building_changes(control).values())
    building_did = (
        statistics.mean(t_buildings) - statistics.mean(c_buildings)
        if t_buildings and c_buildings
        else None
    )

    seed_value = seed + sum(ord(ch) for ch in label)
    ci_low = ci_high = p_value = None
    b_ci_low = b_ci_high = None
    if len(t_buildings) >= 2 and len(c_buildings) >= 2:
        ci_low, ci_high = cluster_bootstrap(t_buildings, c_buildings, seed_value)
        b_ci_low, b_ci_high = cluster_bootstrap(t_buildings, c_buildings, seed_value + 1)
        p_value = permutation_p(t_buildings, c_buildings, seed_value + 2)

    t_slope = within_person_slope([c.person for c in treated], pre, observations)
    c_slope = within_person_slope([c.person for c in control], pre, observations)
    trend_diff = t_slope - c_slope if t_slope is not None and c_slope is not None else None

    lows = []
    highs = []
    for change in treated:
        low, high = missing_bounds(
            change.person, households[change.person], pre, post, observations, cycle, week_index
        )
        if low is not None and high is not None:
            lows.append(low)
            highs.append(high)
    if lows and c_mean is not None:
        bound_low = statistics.mean(lows) - c_mean
        bound_high = statistics.mean(highs) - c_mean
    else:
        bound_low = bound_high = None

    slots = len(pre) + len(post)
    t_obs = sum(c.pre_observed + c.post_observed for c in treated)
    c_obs = sum(c.pre_observed + c.post_observed for c in control)

    return Estimate(
        label=label,
        reference="+".join(sorted(reference)) if reference else "无设施",
        target="+".join(sorted(target)) if target else "无设施",
        floors=floors,
        pre_weeks=list(pre),
        post_weeks=list(post),
        treated_cells=sorted(treated_cells),
        treated_buildings=treated_buildings,
        control_buildings=control_buildings,
        control_audit=audit,
        treated=treated,
        control=control,
        treated_change=t_mean,
        control_change=c_mean,
        did=did,
        unmatched_did=unmatched_did,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        building_did=building_did,
        building_ci_low=b_ci_low,
        building_ci_high=b_ci_high,
        pre_trend_treated=t_slope,
        pre_trend_control=c_slope,
        pre_trend_diff=trend_diff,
        missing_bound_low=bound_low,
        missing_bound_high=bound_high,
        treated_moved=sum(1 for c in treated if c.moved),
        control_moved=sum(1 for c in control if c.moved),
        treated_missing=sum(c.pre_missing + c.post_missing for c in treated),
        control_missing=sum(c.pre_missing + c.post_missing for c in control),
        treated_observed_rate=t_obs / (len(treated) * slots) if treated and slots else 0.0,
        control_observed_rate=c_obs / (len(control) * slots) if control and slots else 0.0,
        cycle=cycle,
        flagged_controls=flagged,
        quarantined=quarantined,
        notes=list(notes),
        confounded=confounded,
        recovery=recovery,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# 开放事件拆分
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    reference: frozenset[str]
    target: frozenset[str]
    cells: list[tuple[str, int]]
    pre: list[str]
    post: list[str]
    removed: frozenset[str] = frozenset()

    @property
    def key(self) -> tuple:
        return (
            tuple(sorted(self.reference)),
            tuple(sorted(self.target)),
            tuple(self.pre),
            tuple(self.post),
            tuple(sorted(self.removed)),
        )


def opening_cohorts(
    availability: Availability,
    opening: Opening,
    max_pre: int = 8,
    max_post: int = 4,
    include_other_projects: bool = False,
) -> list[Cohort]:
    """按 (前组合, 后组合, 前窗口, 后窗口, 同期撤除) 分组。"""
    phase = availability.phase_index[opening.phase_id]
    stop_before = phase["planned"] or opening.week
    grouped: dict[tuple, Cohort] = {}
    for cell in opening.cells:
        reference = opening.reference[cell]
        target = opening.target[cell]
        if opening.kind not in target or opening.kind in reference:
            continue
        removed = frozenset(reference - target)
        pre = pre_window(availability, cell, reference, stop_before, max_pre)
        post = post_window(
            availability,
            cell,
            target,
            opening.week,
            max_post,
            include_other_projects=include_other_projects,
        )
        cohort = Cohort(reference, target, [], pre, post, removed)
        grouped.setdefault(cohort.key, cohort).cells.append(cell)
    return sorted(
        grouped.values(),
        key=lambda c: (sorted(c.reference), sorted(c.target), c.pre[:1], c.post[:1], sorted(c.removed)),
    )


def window_blockers(
    availability: Availability,
    cell: tuple[str, int],
    reference: frozenset[str],
    target: frozenset[str],
    anchor: str,
    max_pre: int = 8,
    max_post: int = 4,
) -> dict:
    """说明前/后窗口为什么没取满，便于会议回查是哪一类周被排除。"""
    building, floor = cell
    ordered = availability.weeks
    if anchor not in ordered:
        anchor = next((w for w in ordered if w >= anchor), ordered[-1])
    start = ordered.index(anchor)

    pre_blocker = ""
    i = start - 1
    taken = 0
    while i >= 0 and taken < max_pre:
        week = ordered[i]
        if not availability.clean_week(building, floor, week, reference):
            pre_blocker = f"{week} {_block_reason(availability, building, floor, week, reference)}"
            break
        taken += 1
        i -= 1

    post_blocker = ""
    i = start
    taken = 0
    while i < len(ordered) and taken < max_post:
        week = ordered[i]
        exposure = availability.exposure(building, floor, week)
        others = availability.other_project_overlap(building, week)
        if exposure.under_construction or exposure.interrupted or others or exposure.kinds != target:
            post_blocker = f"{week} {_block_reason(availability, building, floor, week, target)}"
            break
        taken += 1
        i += 1
    return {"pre_blocker": pre_blocker, "post_blocker": post_blocker}


def _block_reason(
    availability: Availability,
    building: str,
    floor: int,
    week: str,
    expected: frozenset[str],
) -> str:
    exposure = availability.exposure(building, floor, week)
    others = availability.other_project_overlap(building, week)
    parts = []
    if exposure.under_construction:
        parts.append("施工 " + "、".join(exposure.construction_phases))
    if exposure.interrupted:
        parts.append("开放后间断")
    if others:
        parts.append("其他项目 " + "、".join(others))
    if exposure.kinds != expected and not (exposure.under_construction or exposure.interrupted):
        parts.append("组合为 " + "+".join(sorted(exposure.kinds) or ["无"]))
    return "；".join(parts) or "不可比"


def estimate_opening(
    availability: Availability,
    opening: Opening,
    cycle: Cycle,
    seed: int = 20260105,
    contaminated: Mapping[str, str] | None = None,
    contamination=None,
    include_sensitivity: bool = True,
) -> list[Estimate]:
    """对开放事件给出队列估计；窗口不足时另给含其它项目周的敏感性估计。"""
    estimates: list[Estimate] = []
    primary_keys = set()
    for cohort in opening_cohorts(availability, opening):
        key = (cohort.reference, cohort.target, tuple(cohort.cells))
        primary_keys.add(key)
        label = _cohort_label(opening, cohort)
        notes = _window_notes(availability, cohort, opening)
        if cohort.removed:
            notes.append(f"同期撤除 {'+'.join(sorted(cohort.removed))}，不能单独归因")
        if opening.recovery:
            notes.append("恢复开放：组合此前已出现过，非首次接触")
        estimates.append(
            estimate_core(
                availability,
                label,
                cohort.cells,
                cohort.reference,
                cohort.target,
                cohort.pre,
                cohort.post,
                cycle,
                seed=seed,
                contaminated=contaminated,
                contamination=contamination,
                notes=notes,
                confounded=bool(cohort.removed),
                recovery=opening.recovery,
            )
        )

    if include_sensitivity:
        for cohort in opening_cohorts(availability, opening, include_other_projects=True):
            key = (cohort.reference, cohort.target, tuple(cohort.cells))
            if key in primary_keys:
                continue
            if len(cohort.pre) < 2 or len(cohort.post) < 2:
                continue
            notes = ["敏感性：干预期含同周其他社区项目，不能作为主估计"]
            notes += _window_notes(availability, cohort, opening)
            if cohort.removed:
                notes.append(f"同期撤除 {'+'.join(sorted(cohort.removed))}，不能单独归因")
            estimates.append(
                estimate_core(
                    availability,
                    _cohort_label(opening, cohort) + " [含其他项目周]",
                    cohort.cells,
                    cohort.reference,
                    cohort.target,
                    cohort.pre,
                    cohort.post,
                    cycle,
                    seed=seed,
                    contaminated=contaminated,
                    contamination=contamination,
                    notes=notes,
                    confounded=bool(cohort.removed),
                    recovery=opening.recovery,
                    sensitivity=True,
                )
            )
    return estimates


def _window_notes(
    availability: Availability, cohort: Cohort, opening: Opening
) -> list[str]:
    """窗口不足时，指出是哪一类周把窗口截断。"""
    notes = []
    for cell in cohort.cells:
        blockers = window_blockers(
            availability, cell, cohort.reference, cohort.target, opening.week
        )
        prefix = f"{cell[0]} f{cell[1]}"
        if len(cohort.pre) < 2 and blockers["pre_blocker"]:
            notes.append(f"{prefix} 干预前窗口被截断：{blockers['pre_blocker']}")
        if len(cohort.post) < 2 and blockers["post_blocker"]:
            notes.append(f"{prefix} 干预期窗口被截断：{blockers['post_blocker']}")
    # 同一原因只留一条，避免整栋楼重复。
    seen = set()
    unique = []
    for note in notes:
        reason = note.split("：", 1)[-1]
        if reason in seen:
            continue
        seen.add(reason)
        unique.append(note)
    return unique


def _cohort_label(opening: Opening, cohort: Cohort) -> str:
    cells = ", ".join(f"{b} f{f}" for b, f in cohort.cells)
    return (
        f"{opening.phase_id} {('+'.join(sorted(cohort.reference)) or '无设施')}"
        f" → {('+'.join(sorted(cohort.target)) or '无设施')} [{cells}]"
    )


# ---------------------------------------------------------------------------
# 施工与间断
# ---------------------------------------------------------------------------


def period_means(tables: Tables, people: Iterable[str], weeks: Sequence[str]) -> dict:
    observations = observation_lookup(tables)
    values = []
    missing = 0
    moved = 0
    for person in people:
        for week in weeks:
            record = observations.get((person, week))
            if record is None:
                missing += 1
            elif record["state"] == "observed":
                values.append(int(record["outings"]))
            elif record["state"] == "moved":
                moved += 1
            else:
                missing += 1
    return {
        "weeks": list(weeks),
        "mean": statistics.mean(values) if values else None,
        "observed": len(values),
        "missing": missing,
        "moved": moved,
    }


def construction_episode(availability: Availability, phase_id: str) -> dict | None:
    phase = availability.phase_index[phase_id]
    if phase["state"] != "construction":
        return None
    planned = phase["planned"]
    close = phase["close"] or availability.weeks[-1]
    served = availability.served_cells(
        phase["way"], int(phase["first_floor"]), int(phase["last_floor"]), planned
    )
    if not served:
        return None
    buildings = sorted({b for b, _ in served})
    floors = sorted({f for _, f in served})
    people = people_in_cells(availability.tables, served)

    pre = [
        week
        for week in availability.weeks
        if week < planned
        and all(availability.baseline_eligible(b, f, week) for b, f in served)
    ]
    during = [week for week in availability.weeks if planned <= week < close]
    after = [week for week in availability.weeks if week >= close]
    controls = [
        building
        for building in sorted(availability.floors)
        if building not in buildings
        and set(floors) <= set(availability.floors[building])
        and all(
            availability.baseline_eligible(building, f, week)
            for f in floors
            for week in pre + during + after
        )
    ]
    control_people = [
        row["person"]
        for row in availability.tables.households
        if row["building"] in controls and int(row["floor"]) in floors
    ]
    return {
        "phase_id": phase_id,
        "project": phase["project"],
        "way": phase["way"],
        "buildings": buildings,
        "floors": floors,
        "served": served,
        "pre_weeks": pre,
        "during_weeks": during,
        "after_weeks": after,
        "treated": {
            "pre": period_means(availability.tables, people, pre),
            "during": period_means(availability.tables, people, during),
            "after": period_means(availability.tables, people, after),
        },
        "control_buildings": controls,
        "control": {
            "pre": period_means(availability.tables, control_people, pre),
            "during": period_means(availability.tables, control_people, during),
            "after": period_means(availability.tables, control_people, after),
        },
    }


def interruption_episode(availability: Availability, phase_id: str) -> dict | None:
    """登记关闭后，按楼层说明剩余组合、是否恢复、以及各段观察均值。"""
    phase = availability.phase_index[phase_id]
    if not phase["close"]:
        return None
    close = phase["close"]
    served = availability.served_cells(
        phase["way"], int(phase["first_floor"]), int(phase["last_floor"]), phase["actual"]
    )
    rows = []
    for cell in served:
        building, floor = cell
        people = [
            row["person"]
            for row in availability.tables.households
            if row["building"] == building and int(row["floor"]) == floor
        ]
        reopened = [
            week
            for week in availability.weeks
            if week > close and phase["kind"] in availability.combo(building, floor, week)
        ]
        remaining = availability.combo(building, floor, close)
        before = [w for w in availability.weeks if phase["actual"] <= w < close]
        during = [w for w in availability.weeks if close <= w < (reopened[0] if reopened else "9999")]
        rows.append(
            {
                "cell": f"{building} f{floor}",
                "remaining_kinds": "+".join(sorted(remaining)) or "无",
                "reopened_from": reopened[0] if reopened else None,
                "before": period_means(availability.tables, people, before),
                "during": period_means(availability.tables, people, during),
            }
        )
    return {"phase_id": phase_id, "close": close, "rows": rows}


def decompose(
    availability: Availability,
    estimate: Estimate,
    cycle: Cycle,
    contamination=None,
    seed: int = 20260105,
) -> dict:
    """把一个组合估计按楼层带与住户分层拆开，检查改善是否只在某类通道成立。"""
    cells = estimate.treated_cells
    floors = sorted({floor for _, floor in cells})
    bands = {}
    for label, band in _floor_bands(floors):
        subset = [c for c in cells if c[1] in band]
        if not subset:
            continue
        bands[label] = estimate_core(
            availability,
            f"{estimate.label} / 楼层 {label}",
            subset,
            _to_frozen(estimate.reference),
            _to_frozen(estimate.target),
            estimate.pre_weeks,
            estimate.post_weeks,
            cycle,
            seed=seed,
            contamination=contamination,
            notes=list(estimate.notes),
            confounded=estimate.confounded,
            recovery=estimate.recovery,
            sensitivity=estimate.sensitivity,
        )
    groups = {}
    for group in ("ordinary", "mobility_record"):
        if not any(row["group"] == group for row in availability.tables.households):
            continue
        groups[group] = estimate_core(
            availability,
            f"{estimate.label} / {group}",
            cells,
            _to_frozen(estimate.reference),
            _to_frozen(estimate.target),
            estimate.pre_weeks,
            estimate.post_weeks,
            cycle,
            seed=seed,
            contamination=contamination,
            notes=list(estimate.notes),
            confounded=estimate.confounded,
            recovery=estimate.recovery,
            sensitivity=estimate.sensitivity,
            group=group,
        )
    return {"bands": bands, "groups": groups}


def _floor_bands(floors: Sequence[int]) -> list[tuple[str, set[int]]]:
    """按连续楼层切带；一栋楼通常 1–6 层，一条通道的电梯只服务部分楼层。"""
    if not floors:
        return []
    bands: list[tuple[str, set[int]]] = []
    start = previous = floors[0]
    for floor in floors[1:]:
        if floor == previous + 1:
            previous = floor
            continue
        bands.append((_band_label(start, previous), set(range(start, previous + 1))))
        start = previous = floor
    bands.append((_band_label(start, previous), set(range(start, previous + 1))))
    return bands


def _band_label(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def _to_frozen(text: str) -> frozenset[str]:
    if text in ("无设施", ""):
        return frozenset()
    return frozenset(text.split("+"))
