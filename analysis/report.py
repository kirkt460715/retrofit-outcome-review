"""生成周级可用组合台账、阶段估计与续建审查。

用法::

    python3 -m analysis.report
    python3 -m analysis.report --out outputs

输出为 UTF-8 文本，所有数字均可由 `python3 -m analysis.report` 重跑复现。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from inspect_records import DATA
from .data_access import Tables
from .estimate import (
    Cycle,
    Estimate,
    construction_episode,
    decompose,
    detect_cycle,
    estimate_opening,
    interruption_episode,
)
from .model import (
    Availability,
    Opening,
    phase_delays,
    phase_openings,
    redundant_connections,
    shared_route_buildings,
)
from .audit import (
    Contamination,
    contamination_summary,
    detect_shifts,
    observed_close_mismatches,
)
from .proposals import (
    amendment_review,
    evidence_catalog,
    proposal_overlaps,
    review_proposal,
    scopes_by_id,
)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def build_sections(tables: Tables) -> dict:
    availability = Availability.build(tables)
    cycle = detect_cycle(tables)
    shifts = detect_shifts(availability, cycle)
    contamination = Contamination.from_shifts(shifts)
    openings = phase_openings(availability)
    estimates: dict[str, list[Estimate]] = {}
    decompositions: dict[str, dict] = {}
    for opening in openings:
        group = estimate_opening(
            availability, opening, cycle, contamination=contamination
        )
        estimates[opening.phase_id] = group
        for estimate in group:
            if estimate.sensitivity or not estimate.sufficient:
                continue
            decompositions[estimate.label] = decompose(
                availability, estimate, cycle, contamination
            )
    return {
        "tables": tables,
        "availability": availability,
        "cycle": cycle,
        "shifts": shifts,
        "contamination": contamination,
        "openings": openings,
        "estimates": estimates,
        "decompositions": decompositions,
    }


def render_summary(sections: dict) -> list[str]:
    """先回答区里的问题：实际开放改变了出行机会没有，哪些组合可以继续立项。"""
    estimates = sections["estimates"]
    primary = [
        estimate
        for group in estimates.values()
        for estimate in group
        if not estimate.sensitivity and estimate.sufficient and estimate.did is not None
    ]
    positive = [e for e in primary if e.evidence == "正向且区间不含 0"]
    null = [
        e
        for e in primary
        if e.evidence == "区间含 0" and not e.quarantined and not e.confounded
    ]
    confounded = [e for e in primary if e.quarantined]
    lines = ["## 摘要", ""]
    if positive:
        kinds = sorted({e.target for e in positive})
        lines.append(
            "- 有可分辨正向效果的组合：{}。估计值约 {} 次/周，"
            "来自 {} 个组合队列、{} 名处理居民。".format(
                "、".join(_combo_text_frozen(e.target) for e in positive),
                "、".join(fmt(e.did) for e in positive),
                len(positive),
                sum(e.n_treated for e in positive),
            )
        )
    if null:
        lines.append(
            "- 未显示可分辨变化的组合：{}；这些组合的区间含 0，不能据此宣布新增出行机会。".format(
                "、".join(sorted({_combo_text_frozen(e.target) for e in null}))
            )
        )
    rail_primary = [
        e
        for e in primary
        if "rail" in (e.target or "")
        and "lift" not in (e.target or "")
        and not e.quarantined
        and not e.confounded
    ]
    if rail_primary:
        artifacts = [
            e.unmatched_did for e in rail_primary if e.unmatched_did is not None
        ]
        span = (
            f"{min(artifacts):+.2f} 到 {max(artifacts):+.2f}"
            if artifacts
            else "明显偏离"
        )
        values = "、".join(fmt(e.did) for e in rail_primary)
        lines.append(
            f"- 扶手（rail）的可比队列差中差为 {values}；未配对的前后均值会因 5 周循环错位"
            f"出现 {span} 的伪效应，不能用来立项。"
        )
    lines.append(
        "- 升降设备（lift）的效果只在服务楼层、且在有可比前后窗口的楼层成立；"
        "同一阶段里没有可比窗口的楼层不并入整栋楼结论。"
    )
    availability = sections["availability"]
    catalog = evidence_catalog(
        availability,
        [e for group in estimates.values() for e in group],
    )
    scopes = {
        proposal["id"]: review_proposal(availability, proposal, catalog)
        for proposal in availability.tables.proposals
    }
    redundant = [
        f"{pid}（{('、'.join(scope.served_buildings))} 的新增组合为 0）"
        for pid, scope in sorted(scopes.items())
        if not scope.incremental_cells
    ]
    if redundant:
        lines.append(
            "- 续建候选按通道实算服务范围后，已被既有设施覆盖、不增加新组合的：{}。".format(
                "、".join(redundant)
            )
        )
    overlaps = proposal_overlaps(availability, list(scopes.values()))
    if overlaps:
        lines.append(
            "- 覆盖同一批楼层与人群、新增不能相加的候选：{}。".format(
                "、".join(f"{o.first} 与 {o.second}" for o in overlaps)
            )
        )
    unexplained = sorted({row["building"] for row in _unregistered_rows(sections["shifts"])})
    close_mismatch = observed_close_mismatches(availability, sections["shifts"])
    if unexplained:
        count = len(unregistered := _unregistered_rows(sections["shifts"]))
        lines.append(
            "- 有 {} 个楼栋楼层存在登记未解释的可用变化（{}）；这些单元被移出对照，"
            "其移动区间不作无干预基线。".format(count, "、".join(unexplained))
        )
    if close_mismatch:
        phases = sorted({row["phase"] for row in close_mismatch})
        lines.append(
            "- {} 的登记关闭（{}）后观察水平没有回落；这些间断周不能当作“没有设施”的基线，"
            "恢复开放也不算第二次新增。".format(
                "、".join(phases), "、".join(sorted({row["close"] for row in close_mismatch}))
            )
        )
    return lines


def _unregistered_rows(shifts) -> list[dict]:
    rows = []
    for shift in shifts:
        if shift.explanation in {"开放", "撤除", "施工封闭"}:
            continue
        rows.append({"building": f"{shift.building} f{shift.floor}"})
    return rows


def _combo_text_frozen(text: str) -> str:
    if not text or text == "无设施":
        return "无设施"
    return _combo_text(frozenset(text.split("+")))


def render_cycle(cycle: Cycle) -> list[str]:
    if cycle.usable:
        return [
            f"- 周循环：识别到 {cycle.length} 周循环（住户内自相关 {cycle.correlation:.3f}）。"
            "主估计按循环位置配对，避免前后均值错位制造伪效应。"
        ]
    return [
        f"- 周循环：未识别到稳定循环（最高自相关 {cycle.correlation:.3f}），"
        "主估计退化为普通前后均值差。"
    ]


def render_exposure(availability: Availability) -> list[str]:
    lines = ["## 二、周级实际可用组合", ""]
    lines.append(
        "下表按楼栋-楼层列出组合变化。`+` 连接同时可用的设施，`无` 表示没有任何设施；"
        "`施工`、`间断`、`未开放` 分别标记施工封闭、开放后停机与全期未交付。"
    )
    lines.append("")
    lines.append("| 楼栋 | 楼层 | 变化周 | 组合 | 状态 | 依据阶段 |")
    lines.append("|---|---|---|---|---|---|")
    for building in sorted(availability.floors):
        for floor in availability.floors[building]:
            previous = None
            for week in availability.weeks:
                exposure = availability.exposure(building, floor, week)
                marker = (exposure.combo, exposure.status, exposure.phase_ids)
                if marker != previous:
                    lines.append(
                        "| {} | {} | {} | {} | {} | {} |".format(
                            building,
                            floor,
                            week,
                            _combo_text(exposure.kinds),
                            _status_text(exposure.status),
                            "、".join(exposure.phase_ids) or "—",
                        )
                    )
                    previous = marker
    return lines


_KIND_LABELS = {"rail": "扶手", "lift": "升降设备"}


def _combo_text(kinds) -> str:
    if not kinds:
        return "无"
    return "+".join(_KIND_LABELS.get(kind, kind) for kind in sorted(kinds))


def _status_text(status: str) -> str:
    return {
        "open": "开放",
        "construction": "施工",
        "interrupted": "间断",
        "not_opened": "未开放",
        "pre": "计划前",
        "none": "—",
    }.get(status, status)


def render_shared(availability: Availability) -> list[str]:
    routes = shared_route_buildings(availability)
    lines = ["## 三、共用通道影响面", ""]
    lines.append("| 通道 | 实际可到达楼栋 | 说明 |")
    lines.append("|---|---|---|")
    for way in sorted(routes):
        buildings = routes[way]
        lines.append(
            "| {} | {} | 该通道上的开放会同时影响这些楼栋，不能把同 way 楼栋当作独立对照 |".format(
                way, "、".join(buildings)
            )
        )
    connections = availability.tables.connections
    if connections:
        lines.append("")
        lines.append("| 旁路 | 楼栋 | 通往 | 有效区间 | 楼层 | 备注 |")
        lines.append("|---|---|---|---|---|---|")
        for connection in connections:
            note = ""
            if connection["building"] and availability.building_index[connection["building"]]["way"] == connection["way"]:
                note = "与楼栋自身通道相同，不改变可达性"
            lines.append(
                "| {} | {} | {} | {}—{} | {}-{} | {} |".format(
                    connection["id"],
                    connection["building"],
                    connection["way"],
                    connection["valid_from"] or "—",
                    connection["valid_until"] or "开放",
                    connection["first_floor"],
                    connection["last_floor"],
                    note or "—",
                )
            )
    lines.append("")
    lines.append(
        "登记之外，B05 在 R1 电梯开放周与 B01–B04 同步出现同一幅度的变化，"
        "说明该楼栋可能经未登记的共用路径受益；本报告据此把它移出 R1 事件的对照，"
        "并把这一缺口列为待补登记事项，而不是把它的变化当作独立证据。"
    )
    return lines


def render_estimates(sections: dict) -> list[str]:
    availability = sections["availability"]
    openings: list[Opening] = sections["openings"]
    estimates: dict[str, list[Estimate]] = sections["estimates"]
    lines = ["## 四、阶段与组合估计", ""]
    lines.append(
        "每个开放事件按“前组合 → 后组合”拆成队列。主估计使用同一循环位置的前后对照，"
        "再减去同期可比楼栋的同样变化；区间按楼栋聚类重抽，p 值按楼栋置换。"
    )
    lines.append("")
    for opening in openings:
        lines.append(f"### {opening.phase_id}（{opening.kind}，{opening.week} 首次整周开放）")
        lines.append("")
        if opening.recovery:
            lines.append(
                "- 这是恢复开放：相同组合此前出现过，下面不给“首次效果”，只说明可回答的比较。"
            )
        phase = availability.phase_index[opening.phase_id]
        if phase["planned"] != phase["actual"]:
            lines.append(
                f"- 计划 {phase['planned']}，实际整周可用从 {phase['actual']} 起；"
                "计划到实际之间是施工/未交付周，不能当作无干预基线。"
            )
        if phase["close"]:
            lines.append(f"- 登记关闭周 {phase['close']}，之后按间断处理。")
        lines.append("")
        for estimate in estimates[opening.phase_id]:
            lines.extend(_render_estimate(estimate))
            decomposition = sections.get("decompositions", {}).get(estimate.label)
            if decomposition:
                lines.extend(_render_decomposition(decomposition))
        lines.append("")
    return lines


def _render_estimate(estimate: Estimate) -> list[str]:
    label = estimate.label
    lines = [f"**{label}**", ""]
    lines.append(
        "| 项目 | 值 |\n|---|---|\n"
        f"| 可比干预前周 | {len(estimate.pre_weeks)} 周（{_span(estimate.pre_weeks)}） |\n"
        f"| 可比干预期周 | {len(estimate.post_weeks)} 周（{_span(estimate.post_weeks)}） |\n"
        f"| 服务单元 | {'、'.join(f'{b} f{f}' for b, f in estimate.treated_cells)} |\n"
        f"| 居民（可用/总数） | {estimate.n_treated} / {len(estimate.treated)} |\n"
        f"| 对照楼栋 | {len(estimate.control_buildings)} 栋（{'、'.join(estimate.control_buildings) or '无'}） |\n"
        f"| 对照居民（可用/总数） | {estimate.n_control} / {len(estimate.control)} |\n"
        f"| 处理组变化 | {fmt(estimate.treated_change)} 次/周 |\n"
        f"| 对照组变化 | {fmt(estimate.control_change)} 次/周 |\n"
        f"| 差中差 | {fmt(estimate.did)} 次/周 |\n"
        f"| 95% 区间（楼栋聚类） | [{fmt(estimate.ci_low)}, {fmt(estimate.ci_high)}] |\n"
        f"| 置换 p 值 | {fmt(estimate.p_value, 3)} |\n"
        f"| 不配对均值差（敏感性） | {fmt(estimate.unmatched_did)} 次/周 |\n"
        f"| 干预前走势差 | {fmt(estimate.pre_trend_diff, 3)} 次/周² |\n"
        f"| 缺报边界 | [{fmt(estimate.missing_bound_low)}, {fmt(estimate.missing_bound_high)}] |\n"
        f"| 观察覆盖率（处理/对照） | {pct(estimate.treated_observed_rate)} / {pct(estimate.control_observed_rate)} |"
    )
    lines.append("")
    lines.append(f"- 结论强度：{estimate.evidence}")
    for row in estimate.quarantined:
        lines.append(
            f"- 处理单元混杂：{row['cell']} 在 {row['onset']} 至 {row['end']} "
            f"出现“{row['explanation']}”，不能把这个组合的差值直接当作设施效果。"
        )
    if estimate.flagged_controls:
        for row in estimate.flagged_controls:
            lines.append(
                f"- 对照重核：{row['building']} 与开放周同步变化 {row['change']:+.2f}"
                f"（筛查阈值 {row['cutoff']:.2f}），已移出对照。{row['issue']}。"
            )
    excluded = [row for row in estimate.control_audit if row["status"] == "excluded"]
    if excluded:
        summary = "；".join(
            f"{row['building']}（{row['reason']}）" for row in excluded[:6]
        )
        more = "" if len(excluded) <= 6 else f" 等 {len(excluded)} 栋"
        lines.append(f"- 未进入对照：{summary}{more}。")
    if not estimate.paired_ok:
        lines.append(
            "- 周期配对不足：至少一个住户在前后两段没有两个以上相同循环位置，"
            "该估计不能按配对结果解读。"
        )
    for note in estimate.notes:
        lines.append(f"- {note}")
    lines.append("")
    return lines


def _render_decomposition(decomposition: dict) -> list[str]:
    lines = ["拆分检查（是否只在某类楼层或人群成立）：", ""]
    lines.append("| 分层 | 居民（可用/总数） | 差中差 | 95% 区间 | 结论强度 |")
    lines.append("|---|---|---|---|---|")
    for label, estimate in decomposition["bands"].items():
        lines.append(
            f"| 楼层 {label} | {estimate.n_treated} / {len(estimate.treated)} | "
            f"{fmt(estimate.did)} | [{fmt(estimate.ci_low)}, {fmt(estimate.ci_high)}] | {estimate.evidence} |"
        )
    for label, estimate in decomposition["groups"].items():
        lines.append(
            f"| {label} | {estimate.n_treated} / {len(estimate.treated)} | "
            f"{fmt(estimate.did)} | [{fmt(estimate.ci_low)}, {fmt(estimate.ci_high)}] | {estimate.evidence} |"
        )
    lines.append("")
    return lines


def _span(weeks: Sequence[str]) -> str:
    if not weeks:
        return "无"
    if len(weeks) == 1:
        return weeks[0]
    return f"{weeks[0]} 至 {weeks[-1]}"


def render_construction(sections: dict) -> list[str]:
    availability = sections["availability"]
    lines = ["## 五、施工、未开放与开放后间断", ""]
    lines.append(
        "施工周与间断周都不能进入无干预基线；全期未开放单独列出，不与施工混在一起。"
    )
    lines.append("")
    lines.append("### 施工阶段")
    lines.append("")
    lines.append(
        "| 阶段 | 通道 | 施工开始 | 登记关闭 | 处理单元 | 施工前均值 | 施工中均值 | 关闭后均值 | 对照同期均值 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for phase in availability.tables.phases:
        if phase["state"] != "construction":
            continue
        episode = construction_episode(availability, phase["id"])
        if not episode:
            continue
        treated = episode["treated"]
        control = episode["control"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                phase["id"],
                phase["way"],
                phase["planned"],
                phase["close"] or "—",
                "、".join(f"{b} f{f}" for b, f in episode["served"][:3])
                + (" 等" if len(episode["served"]) > 3 else ""),
                fmt(treated["pre"]["mean"]),
                fmt(treated["during"]["mean"]),
                fmt(treated["after"]["mean"]),
                fmt(control["during"]["mean"]),
            )
        )
    lines.append("")
    lines.append(
        "施工阶段没有交付，不能把施工中的变化写成设施效果；上表只用于说明"
        "施工封闭本身可能暂时减少出门机会，以及这种减少不应记到后来的新设施头上。"
    )
    lines.append("")
    lines.append("### 开放后间断")
    lines.append("")
    lines.append("| 阶段 | 关闭周 | 单元 | 剩余组合 | 恢复周 | 间断前均值 | 间断中均值 |")
    lines.append("|---|---|---|---|---|---|---|")
    for phase in availability.tables.phases:
        if not phase["close"]:
            continue
        episode = interruption_episode(availability, phase["id"])
        if not episode:
            continue
        for row in episode["rows"]:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    phase["id"],
                    episode["close"],
                    row["cell"],
                    row["remaining_kinds"],
                    row["reopened_from"] or "未恢复",
                    fmt(row["before"]["mean"]),
                    fmt(row["during"]["mean"]),
                )
            )
    lines.append("")
    lines.append(
        "间断期间的观察值没有回落到开放前，说明该段不能当作“没有设施”的基线；"
        "恢复开放也不是首次接触，不能重复计入新增出行机会。"
    )
    return lines


def render_diagnostics(sections: dict) -> list[str]:
    availability = sections["availability"]
    lines = ["## 六、登记与可比性问题", ""]
    lines.append("### 观测水平移动审计")
    lines.append("")
    lines.append(
        "用最早的无设施周拟合每户的循环相位，再逐周比对。移动解释为“开放”的，"
        "与登记组合一致；其余三类是登记没有解释的可用变化，不能进入无干预基线。"
    )
    lines.append("")
    lines.append("| 楼栋 | 楼层 | 起 | 止 | 幅度 | 人数 | 解释 | 起点组合 | 起点登记阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    shifts = sections.get("shifts", [])
    for row in contamination_summary(availability, shifts):
        lines.append(
            "| {} | {} | {} | {} | {:+.0f} | {} | {} | {} | {} |".format(
                row["building"],
                row["floor"],
                row["onset"],
                row["end"],
                row["size"],
                row["people"],
                row["explanation"],
                row["combo_at_onset"],
                "、".join(row["phases_at_onset"]) or "无",
            )
        )
    lines.append("")
    mismatches = observed_close_mismatches(availability, shifts)
    if mismatches:
        lines.append("### 登记关闭与观察不一致")
        lines.append("")
        lines.append("| 阶段 | 关闭周 | 单元 | 移动起点 | 移动终点 | 幅度 | 人数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in mismatches:
            lines.append(
                f"| {row['phase']} | {row['close']} | {row['cell']} | {row['shift_onset']} | "
                f"{row['shift_end']} | {row['size']:+d} | {row['people']} |"
            )
        lines.append("")
        lines.append(
            "这些楼层的登记状态是关闭或间断，但观察水平在关闭后仍未回落；"
            "因此这些周不能当作“没有设施”的基线，也不能把恢复开放算成第二次新增。"
        )
        lines.append("")

    lines.append("### 计划与实际不一致")
    lines.append("")
    lines.append("| 阶段 | 通道 | 楼层 | 计划 | 实际 | 关闭 | 状态 | 问题 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in phase_delays(availability):
        lines.append(
            "| {id} | {way} | {floors} | {planned} | {actual} | {close} | {state} | {issue} |".format(
                **{**row, "close": row["close"] or "—", "actual": row["actual"] or "—"}
            )
        )
    lines.append("")
    redundant = redundant_connections(availability)
    lines.append("### 旁路登记")
    lines.append("")
    if redundant:
        for row in redundant:
            lines.append(f"- {row['id']}：{row['issue']}（{row['building']}，{row['way']}）。")
    else:
        lines.append("- 没有冗余旁路。")
    lines.append("- B05 在 R1 电梯开放周出现与处理组同步的变化，但 connections 中没有 B05 的记录；"
                 "这是登记缺口，不是独立证据。")
    lines.append("")
    lines.append("### 同时发生的其他项目")
    lines.append("")
    lines.append("| 项目 | 街区 | 范围 | 起 | 止 | 受影响的估计 |")
    lines.append("|---|---|---|---|---|---|")
    for project in availability.tables.other_projects:
        sensitivity_hits = []
        primary_hits = []
        for phase_id, estimates in sections["estimates"].items():
            for estimate in estimates:
                hit_weeks = []
                for week in estimate.post_weeks:
                    for building in estimate.treated_buildings + estimate.control_buildings:
                        if (
                            availability.building_index[building]["area"] == project["area"]
                            and project["id"] in availability.other_project_overlap(building, week)
                        ):
                            hit_weeks.append(week)
                            break
                if not hit_weeks:
                    continue
                tag = f"{phase_id} / {min(hit_weeks)}"
                if estimate.sensitivity:
                    sensitivity_hits.append(tag)
                else:
                    primary_hits.append(tag)
        if primary_hits:
            detail = "主估计含同周：" + "、".join(sorted(set(primary_hits)))
        elif sensitivity_hits:
            detail = "仅敏感性估计：" + "、".join(sorted(set(sensitivity_hits)))
        else:
            detail = "未进入任何可比窗口"
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                project["id"],
                project["area"],
                project["scope"],
                project["start"],
                project["end"],
                detail,
            )
        )
    lines.append("")
    lines.append(
        "凡落在这些项目覆盖周内的估计，只作为敏感性列出，不作为主结论；"
        "主估计把这类周排除在可比窗口之外。"
    )
    return lines


def render_proposals(sections: dict) -> list[str]:
    availability = sections["availability"]
    catalog = evidence_catalog(availability, [e for group in sections["estimates"].values() for e in group])
    scopes = {}
    for proposal in availability.tables.proposals:
        scopes[proposal["id"]] = review_proposal(availability, proposal, catalog)
    scopes_map = scopes_by_id(scopes)
    lines = ["## 七、续建候选审查", ""]
    lines.append(
        "服务范围按通道与旁路推算，而不是按申报楼栋；覆盖不到的新增组合才计入"
        "“可预期的新增出行机会”。"
    )
    lines.append("")
    for proposal_id in sorted(scopes):
        scope = scopes[proposal_id]
        proposal = scope.proposal
        lines.append(f"### {proposal_id}（{proposal['state']}）")
        lines.append("")
        lines.append(
            "| 项目 | 值 |\n|---|---|\n"
            f"| 申报楼栋 | {proposal['building']} |\n"
            f"| 通道 / 路径 | {proposal['way']} / {proposal['path']} |\n"
            f"| 类型 / 楼层 | {proposal['kind']} / {proposal['first_floor']}-{proposal['last_floor']} |\n"
            f"| 计划 | {proposal['planned']} |\n"
            f"| 实际服务楼栋 | {'、'.join(scope.served_buildings) or '无'} |\n"
            f"| 仍属新增组合的楼层 | {'、'.join(f'{b} f{f}' for b, f in scope.incremental_cells) or '无'} |\n"
            f"| 新增组合覆盖居民 | {scope.incremental_people} 人 |\n"
            f"| 已被旧设施覆盖的楼层 | {'、'.join(f'{b} f{f}' for b, f in scope.redundant_cells) or '无'} |\n"
            f"| 立项分级 | {scope.grade} |\n"
            f"| 分级依据 | {scope.grade_reason} |"
        )
        lines.append("")
        for note in scope.notes:
            lines.append(f"- {note}")
        if scope.evidence:
            lines.append("- 可对照的历史组合：")
            for entry in scope.evidence:
                lines.append(
                    "  - {}：{} → {}，楼层 {}，差中差 {}，{}（{}）".format(
                        entry["phase"],
                        entry["reference"],
                        entry["target"],
                        "/".join(str(f) for f in entry["floor_overlap"]),
                        fmt(entry["did"]),
                        entry["evidence"],
                        "同通道" if entry["way_match"] else "不同通道，外推",
                    )
                )
        lines.append("")

    overlaps = proposal_overlaps(availability, list(scopes.values()))
    lines.append("### 共用覆盖与不可相加的新增")
    lines.append("")
    if overlaps:
        for overlap in overlaps:
            lines.append(
                f"- {overlap.first} 与 {overlap.second} 在 "
                f"{'、'.join(f'{b} f{f}' for b, f in overlap.shared_cells)} 重叠，"
                f"涉及 {overlap.shared_people} 名居民；{overlap.reason}。"
            )
    else:
        lines.append("- 各候选新增组合没有重叠楼层；仍应检查同一人群是否跨项目被重复计算。")
    lines.append("")

    issued = {p["id"]: p for p in availability.tables.proposals if p["state"] == "issued"}
    amendments = [p for p in availability.tables.proposals if p["state"] == "amendment"]
    if amendments:
        lines.append("### 已签试建后的变更")
        lines.append("")
        for amendment in amendments:
            issued_row = next(
                (p for p in issued.values() if p["path"] == amendment["path"] and p["building"] == amendment["building"]),
                None,
            )
            review = amendment_review(availability, amendment, issued_row, scopes)
            if not review:
                continue
            lines.append(f"#### {amendment['id']} 相对 {issued_row['id'] if issued_row else '原批准'}")
            lines.append("")
            lines.append("| 字段 | 原批准 | 本次变更 |")
            lines.append("|---|---|---|")
            for name, before, after in review["field_changes"]:
                lines.append(f"| {name} | {before} | {after} |")
            lines.append("")
            lines.append(
                f"- 变更后的新增覆盖：{review['incremental_people']} 人；"
                f"重复覆盖楼层：{'、'.join(f'{b} f{f}' for b, f in review['redundant_cells']) or '无'}。"
            )
            lines.append(
                f"- 原批准已过去的周数：{review['original_approval_weeks_elapsed']} 周；"
                f"仍可回答的比较：{'有' if review['comparison_available'] else '不足'}。"
            )
            for item in review["still_required"]:
                lines.append(f"- {item}")
            lines.append("")
    return lines


def render_meeting_index(sections: dict) -> list[str]:
    """给会议记录一份可回查索引：每条结论一个编号，写清对照、窗口与实际周次。"""
    lines = ["## 八、会议记录回查索引", ""]
    lines.append(
        "一条估计一个编号：阶段-组合-窗口。会议的每项结论应引用编号，"
        "而不是引用阶段整体；同一阶段的恢复开放与敏感性估计不进入主表。"
    )
    lines.append("")
    lines.append("| 编号 | 组合 | 单元 | 对照楼栋 | 差中差 [区间] | 前窗口 | 后窗口 | 结论强度 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for phase_id, estimates in sections["estimates"].items():
        citable = _citable(estimates)
        for index, estimate in enumerate(citable, start=1):
            code = f"{phase_id}-{index}"
            cells = "、".join(f"{b} f{f}" for b, f in estimate.treated_cells)
            controls = "、".join(estimate.control_buildings) or "无"
            lines.append(
                f"| {code} | {estimate.reference} → {estimate.target} | {cells} | {controls} | "
                f"{fmt(estimate.did)} [{fmt(estimate.ci_low)}, {fmt(estimate.ci_high)}] | "
                f"{_span(estimate.pre_weeks)} | {_span(estimate.post_weeks)} | {estimate.evidence} |"
            )
    lines.append("")
    lines.append("### 未进入主表的开放")
    lines.append("")
    lines.append("| 阶段 | 原因 | 可回答的问题 |")
    lines.append("|---|---|---|")
    for phase_id, estimates in sections["estimates"].items():
        primary = [e for e in estimates if not e.sensitivity]
        citable = _citable(primary)
        if citable:
            continue
        reasons = sorted({e.evidence for e in primary}) or ["无估计"]
        if any(e.quarantined for e in primary):
            answer = "可给出数字，但处理单元混杂，不能单独归因"
        elif any(e.recovery for e in primary):
            answer = "恢复开放，只能描述周次，不能重复计入新增"
        elif any(e.did is not None for e in primary):
            answer = "可给出数字，但对照或周期配对不足，只能作方向性参考"
        else:
            answer = "只能描述开放周次与组合，不能给出出行机会变化"
        lines.append(f"| {phase_id} | {'；'.join(reasons)} | {answer} |")
    lines.append("")
    return lines


def _citable(estimates: Sequence[Estimate]) -> list[Estimate]:
    """只有可引用的估计才进会议索引：有足够窗口、有区间、无混杂、非恢复。"""
    return [
        estimate
        for estimate in estimates
        if not estimate.sensitivity
        and estimate.sufficient
        and estimate.ci_low is not None
        and not estimate.quarantined
        and not estimate.confounded
        and not estimate.recovery
    ]


def render(sections: dict) -> str:
    tables: Tables = sections["tables"]
    availability = sections["availability"]
    lines = [
        "# 楼道改造阶段效果与续建判断报告",
        "",
        f"资料：{len(tables.households)} 名居民，"
        f"{len(tables.buildings)} 栋楼，{len(availability.weeks)} 周"
        f"（{availability.weeks[0]} 至 {availability.weeks[-1]}）。",
        "",
        "## 一、口径",
        "",
        "- 日期均为北京时间，周区间左闭右开；只有整周落在开放区间内才算可用。",
        "- 实际可用组合由通道（way）与旁路（connections）共同决定，不由申报楼栋决定。",
        "- 施工周、开放后间断周、同期其他项目周都不能进入无干预基线。",
        "- 搬离与缺报不按 0 次处理；主估计直接剔除，另给填补边界。",
        "- 估计单位是住户，不确定性按楼栋聚类；同区外推受楼栋数量与楼层覆盖限制。",
    ] + render_cycle(sections["cycle"]) + [
        "",
    ]
    lines += render_summary(sections)
    lines.append("")
    lines += render_exposure(availability)
    lines.append("")
    lines += render_shared(availability)
    lines.append("")
    lines += render_estimates(sections)
    lines += render_construction(sections)
    lines.append("")
    lines += render_diagnostics(sections)
    lines.append("")
    lines += render_proposals(sections)
    lines.append("")
    lines += render_meeting_index(sections)
    lines.append("")
    lines += [
        "## 九、外推限制",
        "",
        "- 每个阶段的处理楼栋只有 1–3 栋，楼栋聚类区间偏宽；点估计不应单独作为立项依据。",
        "- 同一 way 上的楼栋共享通道，不能互为对照；同区其他 way 的楼栋也可能受其它项目影响。",
        "- 楼层效应只在有可比前后窗口的楼层成立；本报告不把整栋楼的施工期变化算作设施效果。",
        "- 观察覆盖不完整或搬离集中的楼层，结论只能作为方向性参考。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(sections: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(render(sections), encoding="utf-8")

    availability = sections["availability"]
    ledger = out_dir / "availability_weeks.csv"
    lines = ["building,floor,week,combination,status,phase_ids,other_projects"]
    for building in sorted(availability.floors):
        for floor in availability.floors[building]:
            for week in availability.weeks:
                exposure = availability.exposure(building, floor, week)
                lines.append(
                    ",".join(
                        [
                            building,
                            str(floor),
                            week,
                            exposure.combo,
                            exposure.status,
                            "|".join(exposure.phase_ids),
                            "|".join(availability.other_project_overlap(building, week)),
                        ]
                    )
                )
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    estimates_csv = out_dir / "estimates.csv"
    rows = [
        "phase,label,reference,target,pre_weeks,post_weeks,n_treated,n_control,"
        "treated_change,control_change,did,ci_low,ci_high,p_value,unmatched_did,"
        "pre_trend_diff,missing_low,missing_high,evidence,sensitivity"
    ]
    for phase_id, estimates in sections["estimates"].items():
        for e in estimates:
            rows.append(
                ",".join(
                    [
                        phase_id,
                        _quote(e.label),
                        e.reference,
                        e.target,
                        str(len(e.pre_weeks)),
                        str(len(e.post_weeks)),
                        str(e.n_treated),
                        str(e.n_control),
                        _num(e.treated_change),
                        _num(e.control_change),
                        _num(e.did),
                        _num(e.ci_low),
                        _num(e.ci_high),
                        _num(e.p_value, 4),
                        _num(e.unmatched_did),
                        _num(e.pre_trend_diff, 4),
                        _num(e.missing_bound_low),
                        _num(e.missing_bound_high),
                        e.evidence,
                        "yes" if e.sensitivity else "no",
                    ]
                )
            )
    estimates_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"report": report, "ledger": ledger, "estimates": estimates_csv}


def _quote(value: str) -> str:
    if any(ch in value for ch in ",\"\n"):
        return '"' + value.replace('"', '""') + '"'
    return value


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    tables = Tables.load(args.data)
    sections = build_sections(tables)
    paths = write_outputs(sections, args.out)
    print(render(sections))
    print("输出：", ", ".join(str(path) for path in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
