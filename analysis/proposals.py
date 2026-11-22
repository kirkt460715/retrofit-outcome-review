"""续建候选：服务范围、覆盖增量、共用通道去重与证据匹配。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .estimate import Estimate
from .model import Availability

# 证据分级用词，直接进入立项说明。
EXPECTED = "可按现有证据预期"
OBSERVE = "只适合先行观察"


@dataclass
class ProposalScope:
    proposal: Mapping[str, str]
    planned_week: str
    served_cells: list[tuple[str, int]]
    served_buildings: list[str]
    incremental_cells: list[tuple[str, int]]
    incremental_people: int
    incremental_buildings: list[str]
    existing_provider: dict[tuple[str, int], list[str]]
    redundant_cells: list[tuple[str, int]]
    grade: str
    grade_reason: str
    evidence: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ProposalOverlap:
    first: str
    second: str
    shared_cells: list[tuple[str, int]]
    shared_people: int
    reason: str


def _planned_week(proposal: Mapping[str, str], weeks: Sequence[str]) -> str:
    planned = proposal["planned"]
    if not planned:
        return weeks[-1]
    if planned in weeks:
        return planned
    later = [w for w in weeks if w >= planned]
    if later:
        return later[0]
    return weeks[-1]


def _served(
    availability: Availability,
    way: str,
    first_floor: int,
    last_floor: int,
    week: str,
    building: str | None = None,
) -> list[tuple[str, int]]:
    """设施在服务范围内实际能触达的楼栋-楼层。

    楼栋标签只是申报单位，真正的服务范围由 way 与旁路决定。
    """
    out = []
    for candidate in sorted(availability.floors):
        if building is not None and candidate != building:
            continue
        for floor in availability.floors[candidate]:
            if not (first_floor <= floor <= last_floor):
                continue
            if availability.reaches(candidate, way, floor, week):
                out.append((candidate, floor))
    return out


def _current_providers(
    availability: Availability, cell: tuple[str, int], kind: str, week: str
) -> list[str]:
    """当前已经提供该 kind 的阶段，用于识别旧设施与重复覆盖。"""
    building, floor = cell
    providers = []
    for phase in availability.tables.phases:
        if phase["kind"] != kind:
            continue
        if phase["state"] == "construction" or not phase["actual"]:
            continue
        if not (int(phase["first_floor"]) <= floor <= int(phase["last_floor"])):
            continue
        if not availability.reaches(building, phase["way"], floor, week):
            continue
        if availability.exposure(building, floor, week).status != "open":
            continue
        if kind in availability.combo(building, floor, week):
            providers.append(phase["id"])
    return providers


def _people(availability: Availability, cells: Iterable[tuple[str, int]]) -> int:
    wanted = set(cells)
    return sum(
        1
        for row in availability.tables.households
        if (row["building"], int(row["floor"])) in wanted
    )


def evidence_catalog(
    availability: Availability, estimates: Sequence[Estimate]
) -> list[dict]:
    """把历史估计整理成可匹配的证据条目。"""
    catalog = []
    for estimate in estimates:
        if estimate.sensitivity:
            continue
        phase_id = estimate.label.split()[0] if estimate.label else ""
        catalog.append(
            {
                "phase": phase_id,
                "reference": estimate.reference,
                "target": estimate.target,
                "floors": estimate.floors,
                "did": estimate.did,
                "evidence": estimate.evidence,
                "sufficient": estimate.sufficient,
                "buildings": estimate.treated_buildings,
                "baseline_groups": sorted(
                    {
                        availability.building_index[building]["baseline_group"]
                        for building in estimate.treated_buildings
                    }
                ),
                "pre_weeks": len(estimate.pre_weeks),
                "post_weeks": len(estimate.post_weeks),
                "confounded": estimate.confounded,
                "recovery": estimate.recovery,
            }
        )
    return catalog


def review_proposal(
    availability: Availability,
    proposal: Mapping[str, str],
    catalog: Sequence[dict],
) -> ProposalScope:
    weeks = availability.weeks
    week = _planned_week(proposal, weeks)
    first, last = int(proposal["first_floor"]), int(proposal["last_floor"])
    # 服务范围必须按通道实算：申报楼栋只是立项单位，通道上的其它楼栋同样会被覆盖。
    served = _served(availability, proposal["way"], first, last, week)
    buildings = sorted({b for b, _ in served})

    incremental = []
    redundant = []
    providers: dict[tuple[str, int], list[str]] = {}
    for cell in served:
        existing = _current_providers(availability, cell, proposal["kind"], weeks[-1])
        if existing:
            providers[cell] = existing
            redundant.append(cell)
        else:
            incremental.append(cell)

    notes = []
    declared = {proposal["building"]}
    if set(buildings) - declared:
        notes.append(
            "申请写的是 {}，但设施在 {} 上，实际还会覆盖 {} 的相同楼层".format(
                proposal["building"],
                proposal["way"],
                "、".join(sorted(set(buildings) - declared)),
            )
        )
    if redundant:
        notes.append(
            "{} 个楼层已由 {} 提供同类设施，不增加可用组合".format(
                len(redundant), "、".join(sorted({p for v in providers.values() for p in v}))
            )
        )

    grade, reason, matched = _grade(proposal, catalog, availability)
    return ProposalScope(
        proposal=proposal,
        planned_week=week,
        served_cells=served,
        served_buildings=buildings,
        incremental_cells=incremental,
        incremental_people=_people(availability, incremental),
        incremental_buildings=sorted({b for b, _ in incremental}),
        existing_provider=providers,
        redundant_cells=redundant,
        grade=grade,
        grade_reason=reason,
        evidence=matched,
        notes=notes,
    )


def _grade(
    proposal: Mapping[str, str],
    catalog: Sequence[dict],
    availability: Availability,
) -> tuple[str, str, list[dict]]:
    """按 way、楼层、组合与人群匹配历史中较有依据的开放。"""
    kind = proposal["kind"]
    way = proposal["way"]
    floors = set(range(int(proposal["first_floor"]), int(proposal["last_floor"]) + 1))
    group = availability.building_index[proposal["building"]]["baseline_group"]

    matched = []
    for entry in catalog:
        if kind not in entry["target"] or kind in entry["reference"]:
            continue
        if entry["confounded"] or entry["recovery"]:
            continue
        overlap = sorted(floors & set(entry["floors"]))
        if not overlap:
            continue
        entry = dict(entry)
        entry["floor_overlap"] = overlap
        entry["way_match"] = bool(entry["buildings"]) and all(
            availability.building_index[b]["way"] == way for b in entry["buildings"]
        )
        matched.append(entry)

    if not matched:
        return OBSERVE, "没有同一设施类型的历史开放可供对照", []

    usable = [e for e in matched if e["evidence"] == "正向且区间不含 0" and e["sufficient"]]
    if usable:
        strong = [
            e
            for e in usable
            if e["way_match"] and group in e["baseline_groups"]
        ]
        if strong:
            return EXPECTED, "同一通道、同一楼层带与同一立项前需求组有正向可比证据", matched
        return (
            OBSERVE,
            "有正向证据但通道或立项前需求组不同，属于外推，只适合先行观察",
            matched,
        )
    return OBSERVE, "历史上同类设施未显示可分辨的出行机会变化", matched


def proposal_overlaps(
    availability: Availability, scopes: Sequence[ProposalScope]
) -> list[ProposalOverlap]:
    """找出不同候选覆盖同一批人、同一批楼层的部分。"""
    out = []
    for i, first in enumerate(scopes):
        for second in scopes[i + 1 :]:
            if first.proposal["id"] == second.proposal["id"]:
                continue
            if first.proposal["kind"] != second.proposal["kind"]:
                # 不同设施类型只有在同一 way、同一楼层时才会覆盖同一组合；
                # 这里仍列出，供判断是否属于同一人群。
                pass
            shared = sorted(set(first.incremental_cells) & set(second.incremental_cells))
            if not shared:
                continue
            out.append(
                ProposalOverlap(
                    first=first.proposal["id"],
                    second=second.proposal["id"],
                    shared_cells=shared,
                    shared_people=_people(availability, shared),
                    reason="两个单项覆盖相同楼栋楼层，宣布的新增出行机会不能相加",
                )
            )
    return out


def amendment_review(
    availability: Availability,
    proposal: Mapping[str, str],
    issued: Mapping[str, str] | None,
    scopes: Mapping[str, ProposalScope],
) -> dict:
    """已签试建后来变更时，列出仍要履行的工程承诺与可回答的比较。"""
    if issued is None:
        return {}
    changes = []
    for field_name in ("way", "path", "kind", "first_floor", "last_floor", "planned", "building"):
        if issued[field_name] != proposal[field_name]:
            changes.append((field_name, issued[field_name], proposal[field_name]))
    scope = scopes[proposal["id"]]
    issued_weeks = [w for w in availability.weeks if w >= issued["planned"]]
    return {
        "issued": issued,
        "amendment": proposal,
        "field_changes": changes,
        "incremental_people": scope.incremental_people,
        "redundant_cells": scope.redundant_cells,
        "original_approval_weeks_elapsed": len(issued_weeks),
        "comparison_available": bool(scope.incremental_cells) and scope.grade != OBSERVE
        and not scope.redundant_cells,
        "still_required": [
            "按原批准继续履行的工程承诺需由项目组逐项勾销",
            "变更后的试建只能回答其实际开放范围与干净对照周内的比较",
            "原批准覆盖但本次变更未施工的楼层，需要补登记或补充观察",
        ],
        "notes": scope.notes,
    }


def scopes_by_id(scopes: Mapping[str, ProposalScope]) -> dict[str, ProposalScope]:
    return dict(scopes)
