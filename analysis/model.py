"""按周推算谁在什么时候接触哪种设施组合。

设计范围（phases）与实际通过路线（connections）分开登记：
- 一个设施阶段只服务它自己的 way；
- 楼栋只有在自己的 way 上，或存在当周有效、楼层覆盖的旁路时，才算接触到该 way；
- 只有整周都落在开放区间内才算整周可用，半周不进入暴露口径；
- 施工、计划未开工、开放后间断、全期未开放分开标记。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .data_access import Tables, index, parse_date, week_end

OPEN = "open"
CONSTRUCTION = "construction"
INTERRUPTED = "interrupted"
PRE = "pre"
NOT_OPENED = "not_opened"
NONE = "none"

# 可以当作“无设施”观察的状态；施工与开放后间断一律不可以。
BASELINE_STATUSES = {NONE, PRE, NOT_OPENED}


def phase_status(phase: Mapping[str, str], week: str) -> str:
    """返回阶段在该周的登记状态，日期按左闭右开解释。

    planned 与 actual 之间的周属于施工/未交付，不能当作无干预基线；
    actual 之后 close 之前的周为开放；close 之后为开放后间断；
    从未有 actual 的阶段为全期未开放。
    """
    planned = phase["planned"]
    actual = phase["actual"]
    close = phase["close"]

    if phase["state"] == CONSTRUCTION:
        if planned and week >= planned and (not close or week < close):
            return CONSTRUCTION
        return NOT_OPENED
    if not actual:
        if planned and week >= planned:
            return NOT_OPENED
        return PRE
    if planned and week < planned:
        return PRE
    if week < actual:
        return CONSTRUCTION
    if close and week >= close:
        return INTERRUPTED
    return OPEN


@dataclass(frozen=True)
class Exposure:
    """一个楼栋-楼层-周的实际可用组合。"""

    building: str
    floor: int
    week: str
    kinds: frozenset[str]
    ways: frozenset[str]
    phase_ids: tuple[str, ...]
    status: str
    under_construction: bool
    interrupted: bool
    construction_phases: tuple[str, ...]

    @property
    def combo(self) -> str:
        return "+".join(sorted(self.kinds)) if self.kinds else NONE


@dataclass
class Availability:
    """暴露时间表与检索辅助。"""

    tables: Tables
    weeks: list[str]
    floors: dict[str, list[int]]
    exposures: dict[tuple[str, int, str], Exposure]
    building_index: dict[str, Mapping[str, str]]
    phase_index: dict[str, Mapping[str, str]]
    ways: list[str]

    @classmethod
    def build(cls, tables: Tables) -> "Availability":
        weeks = tables.weeks()
        buildings = index(tables.buildings, "building")
        floors: dict[str, list[int]] = defaultdict(list)
        for household in tables.households:
            floors[household["building"]].append(int(household["floor"]))
        floors = {b: sorted(set(fs)) for b, fs in floors.items()}

        availability = cls(
            tables=tables,
            weeks=weeks,
            floors=floors,
            exposures={},
            building_index=buildings,
            phase_index=index(tables.phases, "id"),
            ways=sorted({phase["way"] for phase in tables.phases}),
        )
        for week in weeks:
            for building in sorted(floors):
                for floor in floors[building]:
                    availability.exposures[(building, floor, week)] = availability._cell(
                        building, floor, week
                    )
        return availability

    # --- 路线可达性 -------------------------------------------------
    def connection_active(self, connection: Mapping[str, str], week: str) -> bool:
        """整周都落在旁路有效期内才计为可用。"""
        if not connection["valid_from"]:
            return False
        if parse_date(connection["valid_from"]) > parse_date(week):
            return False
        if connection["valid_until"] and week_end(week) > parse_date(connection["valid_until"]):
            return False
        return True

    def reaches(self, building: str, way: str, floor: int, week: str) -> bool:
        """楼栋是否能在该周通过自己的 way 或旁路到达给定 way。"""
        if self.building_index[building]["way"] == way:
            return True
        for connection in self.tables.connections:
            if connection["building"] != building or connection["way"] != way:
                continue
            if not (int(connection["first_floor"]) <= floor <= int(connection["last_floor"])):
                continue
            if self.connection_active(connection, week):
                return True
        return False

    def _cell(self, building: str, floor: int, week: str) -> Exposure:
        kinds: set[str] = set()
        ways: set[str] = set()
        phase_ids: list[str] = []
        constructions: list[str] = []
        interrupted = False
        not_opened = False
        pending = False

        for phase in self.tables.phases:
            if not (int(phase["first_floor"]) <= floor <= int(phase["last_floor"])):
                continue
            if not self.reaches(building, phase["way"], floor, week):
                continue
            status = phase_status(phase, week)
            if status == OPEN:
                kinds.add(phase["kind"])
                ways.add(phase["way"])
                phase_ids.append(phase["id"])
            elif status == CONSTRUCTION:
                constructions.append(phase["id"])
            elif status == INTERRUPTED:
                interrupted = True
                phase_ids.append(phase["id"])
            elif status == NOT_OPENED:
                not_opened = True
            elif status == PRE:
                pending = True

        if kinds:
            status = OPEN
        elif interrupted:
            status = INTERRUPTED
        elif constructions:
            status = CONSTRUCTION
        elif not_opened:
            status = NOT_OPENED
        elif pending:
            status = PRE
        else:
            status = NONE

        return Exposure(
            building=building,
            floor=floor,
            week=week,
            kinds=frozenset(kinds),
            ways=frozenset(ways),
            phase_ids=tuple(sorted(phase_ids)),
            status=status,
            under_construction=bool(constructions),
            interrupted=interrupted,
            construction_phases=tuple(sorted(constructions)),
        )

    def exposure(self, building: str, floor: int, week: str) -> Exposure:
        return self.exposures[(building, floor, week)]

    def combo(self, building: str, floor: int, week: str) -> frozenset[str]:
        return self.exposure(building, floor, week).kinds

    # --- 其他项目与基线资格 ----------------------------------------
    def other_project_overlap(self, building: str, week: str) -> list[str]:
        """同周同时开启、且与该楼栋所在街区重叠的其他项目。"""
        area = self.building_index[building]["area"]
        start, end = parse_date(week), week_end(week)
        hits = []
        for project in self.tables.other_projects:
            if project["area"] != area:
                continue
            p_start = parse_date(project["start"])
            p_end = parse_date(project["end"])
            if start < p_end and p_start < end:
                hits.append(project["id"])
        return hits

    def baseline_eligible(self, building: str, floor: int, week: str) -> bool:
        """能否把该周当作该楼栋-楼层的无设施观察。

        已有任何设施、正在施工、开放后间断、或同周有其它社区项目，都不能
        当作无干预基线。全期未开放但施工已结束的周可以，因其没有设施也没有
        在建工程；这一宽松处理会在报告中标注。
        """
        exposure = self.exposure(building, floor, week)
        if exposure.kinds:
            return False
        if exposure.status not in BASELINE_STATUSES:
            return False
        if exposure.under_construction:
            return False
        if self.other_project_overlap(building, week):
            return False
        return True

    def clean_week(self, building: str, floor: int, week: str, kinds: frozenset[str]) -> bool:
        """组合恰为目标、且没有施工/间断/其它项目的可比周。"""
        exposure = self.exposure(building, floor, week)
        return (
            exposure.kinds == kinds
            and not exposure.under_construction
            and not exposure.interrupted
            and not self.other_project_overlap(building, week)
        )

    def served_cells(
        self,
        way: str,
        first_floor: int,
        last_floor: int,
        week: str,
        kinds: Iterable[str] = (),
    ) -> list[tuple[str, int]]:
        """服务范围：谁在该周能通过自己的 way 或旁路接触到这个 way。"""
        cells = []
        for building in sorted(self.floors):
            for floor in self.floors[building]:
                if not (first_floor <= floor <= last_floor):
                    continue
                if not self.reaches(building, way, floor, week):
                    continue
                if kinds and not set(kinds) <= set(self.combo(building, floor, week)):
                    continue
                cells.append((building, floor))
        return cells


# ---------------------------------------------------------------------------
# 事件发现
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Opening:
    """一个设施阶段首次（或恢复）整周可用的时点。"""

    phase_id: str
    project: str
    way: str
    kind: str
    week: str
    cells: tuple[tuple[str, int], ...]
    reference: dict[tuple[str, int], frozenset[str]]
    target: dict[tuple[str, int], frozenset[str]]
    recovery: bool

    @property
    def id(self) -> str:
        return f"OPEN-{self.phase_id}"


def phase_openings(availability: Availability) -> list[Opening]:
    """对每个有 actual 的阶段，找出其首次整周开放的周与受影响单元。"""
    openings = []
    for phase in availability.tables.phases:
        if phase["state"] == "construction" or not phase["actual"]:
            continue
        anchor = phase["actual"]
        if anchor not in availability.weeks:
            later = [w for w in availability.weeks if w >= anchor]
            if not later:
                continue
            anchor = later[0]
        prior = [w for w in availability.weeks if w < anchor]
        previous_week = prior[-1] if prior else None

        cells = []
        reference = {}
        target = {}
        recovery = False
        for building in sorted(availability.floors):
            for floor in availability.floors[building]:
                if not (int(phase["first_floor"]) <= floor <= int(phase["last_floor"])):
                    continue
                if not availability.reaches(building, phase["way"], floor, anchor):
                    continue
                now = availability.exposure(building, floor, anchor)
                if phase["id"] not in now.phase_ids or phase["kind"] not in now.kinds:
                    continue
                if previous_week is None:
                    before = frozenset()
                else:
                    before = availability.combo(building, floor, previous_week)
                    previous = availability.exposure(building, floor, previous_week)
                    if previous.interrupted or phase["id"] in previous.phase_ids:
                        recovery = True
                if now.kinds == before:
                    continue
                cells.append((building, floor))
                reference[(building, floor)] = before
                target[(building, floor)] = now.kinds

        if not cells:
            continue
        openings.append(
            Opening(
                phase_id=phase["id"],
                project=phase["project"],
                way=phase["way"],
                kind=phase["kind"],
                week=anchor,
                cells=tuple(sorted(cells)),
                reference=reference,
                target=target,
                recovery=recovery,
            )
        )
    return openings


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def redundant_connections(availability: Availability) -> list[dict]:
    """登记到楼栋自身 way 的旁路不改变可达性。"""
    out = []
    for connection in availability.tables.connections:
        building = connection["building"]
        if availability.building_index[building]["way"] == connection["way"]:
            out.append(
                {
                    "id": connection["id"],
                    "building": building,
                    "way": connection["way"],
                    "issue": "旁路与楼栋自身 way 相同，不改变可达楼层",
                }
            )
    return out


def phase_delays(availability: Availability) -> list[dict]:
    """计划与实际不一致、全期未开放、开放后间断的登记情况。"""
    out = []
    for phase in availability.tables.phases:
        issues = []
        if phase["state"] == CONSTRUCTION:
            issues.append("施工后未交付，全期未开放")
        elif not phase["actual"]:
            issues.append("无实际开放日期")
        elif phase["actual"] != phase["planned"]:
            issues.append("实际开放晚于计划")
        if phase["close"] and phase["state"] != CONSTRUCTION:
            issues.append("开放后登记间断")
        if issues:
            out.append(
                {
                    "id": phase["id"],
                    "project": phase["project"],
                    "way": phase["way"],
                    "kind": phase["kind"],
                    "floors": f'{phase["first_floor"]}-{phase["last_floor"]}',
                    "planned": phase["planned"],
                    "actual": phase["actual"],
                    "close": phase["close"],
                    "state": phase["state"],
                    "issue": "；".join(issues),
                }
            )
    return out


def shared_route_buildings(availability: Availability) -> dict[str, list[str]]:
    """每个 way 上实际可到达的楼栋，用于共享通道影响面。"""
    routes: dict[str, list[str]] = defaultdict(list)
    last = availability.weeks[-1]
    for building in sorted(availability.floors):
        for way in availability.ways:
            if any(
                availability.reaches(building, way, floor, last)
                for floor in availability.floors[building]
            ):
                routes[way].append(building)
    return dict(routes)


