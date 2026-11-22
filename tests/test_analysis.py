"""检查可用性口径、对照重核、周期配对与续建审查。"""
import tempfile
import unittest
from pathlib import Path

from analysis.audit import Contamination, detect_shifts, observed_close_mismatches
from analysis.data_access import Tables
from analysis.estimate import (
    detect_cycle,
    estimate_opening,
    person_change,
    pre_window,
)
from analysis.model import Availability, phase_openings, phase_status
from analysis.proposals import proposal_overlaps, review_proposal, evidence_catalog
from analysis.report import build_sections, render
from inspect_records import DATA


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = Tables.load(DATA)
        cls.availability = Availability.build(cls.tables)
        cls.cycle = detect_cycle(cls.tables)
        cls.shifts = detect_shifts(cls.availability, cls.cycle)
        cls.contamination = Contamination.from_shifts(cls.shifts)
        cls.openings = {opening.phase_id: opening for opening in phase_openings(cls.availability)}

    # --- 可用性口径 -------------------------------------------------
    def test_week_is_half_open_and_whole_week_only(self):
        phase = self.availability.phase_index["E01"]
        # 计划前一周不可用，实际开放当周整周可用。
        self.assertEqual(phase_status(phase, "2026-01-26"), "pre")
        self.assertEqual(phase_status(phase, "2026-02-02"), "construction")
        self.assertEqual(phase_status(phase, "2026-03-02"), "open")

    def test_connection_extends_reach(self):
        # B04 自身是 R2，经 C01 才能在 2026-04-06 起接触 R1 电梯。
        self.assertFalse(
            self.availability.reaches("B04", "R1", 3, "2026-03-30")
        )
        self.assertTrue(
            self.availability.reaches("B04", "R1", 3, "2026-04-06")
        )
        # 旁路只覆盖 2–6 层。
        self.assertFalse(
            self.availability.reaches("B04", "R1", 1, "2026-04-06")
        )

    def test_phase_kind_only_serves_its_way(self):
        # B05 在 R2，不能在登记上接触 R1。
        self.assertFalse(self.availability.reaches("B05", "R1", 3, "2026-04-06"))

    def test_construction_is_not_baseline(self):
        self.assertFalse(
            self.availability.baseline_eligible("B01", 1, "2026-02-02")
        )
        self.assertTrue(
            self.availability.baseline_eligible("B06", 1, "2026-02-02")
        )

    def test_other_project_week_is_not_baseline(self):
        # D2 的 X01 覆盖 2026-04-13 起，B01 属于 D2。
        self.assertFalse(
            self.availability.baseline_eligible("B01", 1, "2026-04-13")
        )

    def test_unopened_phase_is_distinct_from_construction(self):
        phase = self.availability.phase_index["E06"]
        self.assertEqual(phase_status(phase, "2026-03-23"), "construction")
        self.assertEqual(phase_status(phase, "2026-04-13"), "not_opened")

    # --- 周期配对 ---------------------------------------------------
    def test_cycle_is_detected(self):
        self.assertTrue(self.cycle.usable)
        self.assertEqual(self.cycle.length, 5)

    def test_moved_and_missing_are_not_zero(self):
        observations = {(row["person"], row["week"]): row for row in self.tables.observations}
        week_index = {week: i for i, week in enumerate(self.availability.weeks)}
        row = next(r for r in self.tables.households if r["person"] == "H000")
        change = person_change(
            "H000",
            row,
            ["2026-01-05", "2026-01-12"],
            ["2026-06-08", "2026-06-15"],
            observations,
            self.cycle,
            week_index,
        )
        self.assertIsNone(change.change)
        self.assertTrue(change.moved)
        self.assertEqual(change.post_missing, 0)

    def test_cycle_pairing_removes_phase_artifact(self):
        # E01 的扶手在所有楼层都没有真实变化；不配对均值差会出现约 -0.47 的伪效应。
        estimates = estimate_opening(
            self.availability,
            self.openings["E01"],
            self.cycle,
            contamination=self.contamination,
        )
        primary = [e for e in estimates if not e.sensitivity and e.sufficient]
        self.assertTrue(primary)
        for estimate in primary:
            self.assertAlmostEqual(estimate.did, 0.0)
            if estimate.unmatched_did is not None:
                self.assertGreater(abs(estimate.unmatched_did), 0.2)

    # --- 对照重核 ---------------------------------------------------
    def test_b05_is_quarantined_as_unregistered_route(self):
        b05 = [
            shift for shift in self.shifts if shift.building == "B05" and shift.explanation != "开放"
        ]
        self.assertTrue(b05)
        for shift in b05:
            self.assertIn(shift.explanation, {"未登记的实际可用变化"})

    def test_contaminated_building_not_used_as_control(self):
        estimates = estimate_opening(
            self.availability,
            self.openings["E02"],
            self.cycle,
            contamination=self.contamination,
        )
        for estimate in estimates:
            if not estimate.post_weeks:
                continue
            # B05 自 2026-04-06 起有登记未解释的 +1，不能当任何 R1 组合的对照。
            self.assertNotIn("B05", estimate.control_buildings)

    def test_registered_close_does_not_reset_exposure(self):
        mismatches = observed_close_mismatches(self.availability, self.shifts)
        self.assertTrue(mismatches)
        self.assertTrue(all(row["phase"] == "E02" for row in mismatches))

    def test_lift_effect_only_on_served_floors(self):
        estimates = estimate_opening(
            self.availability,
            self.openings["E02"],
            self.cycle,
            contamination=self.contamination,
        )
        primary = [e for e in estimates if not e.sensitivity and e.sufficient]
        self.assertTrue(primary)
        self.assertTrue(all(estimate.did > 0.5 for estimate in primary))
        # E01 扶手没有可分辨效果。
        rail = [
            e
            for e in estimate_opening(
                self.availability,
                self.openings["E01"],
                self.cycle,
                contamination=self.contamination,
            )
            if not e.sensitivity
        ]
        self.assertTrue(all(abs(e.did or 0.0) < 1e-9 for e in rail))

    # --- 续建 -------------------------------------------------------
    def test_proposal_scope_follows_route_not_label(self):
        sections = build_sections(self.tables)
        catalog = evidence_catalog(
            self.availability,
            [e for group in sections["estimates"].values() for e in group],
        )
        scope = review_proposal(
            self.availability,
            next(p for p in self.tables.proposals if p["id"] == "N03"),
            catalog,
        )
        self.assertEqual(scope.served_buildings, ["B12", "B13", "B14", "B15"])
        self.assertEqual(len(scope.incremental_cells), 16)

    def test_issued_rail_is_redundant(self):
        sections = build_sections(self.tables)
        catalog = evidence_catalog(
            self.availability,
            [e for group in sections["estimates"].values() for e in group],
        )
        scope = review_proposal(
            self.availability,
            next(p for p in self.tables.proposals if p["id"] == "N01"),
            catalog,
        )
        self.assertEqual(scope.incremental_cells, [])
        self.assertTrue(scope.redundant_cells)

    def test_overlapping_proposals_are_not_additive(self):
        sections = build_sections(self.tables)
        catalog = evidence_catalog(
            self.availability,
            [e for group in sections["estimates"].values() for e in group],
        )
        scopes = [
            review_proposal(self.availability, proposal, catalog)
            for proposal in self.tables.proposals
        ]
        overlaps = proposal_overlaps(self.availability, scopes)
        pairs = {(row.first, row.second) for row in overlaps}
        self.assertIn(("N02", "N04"), pairs)

    # --- 报告 -------------------------------------------------------
    def test_report_is_reproducible(self):
        first = render(build_sections(self.tables))
        second = render(build_sections(self.tables))
        self.assertEqual(first, second)
        self.assertIn("会议记录回查索引", first)
        self.assertIn("E02-1", first)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = Tables.load(DATA)
        cls.availability = Availability.build(cls.tables)

    def test_pre_window_stops_at_construction(self):
        weeks = pre_window(
            self.availability, ("B01", 3), frozenset({"rail"}), "2026-03-23"
        )
        self.assertEqual(weeks[-1], "2026-03-16")
        self.assertNotIn("2026-03-23", weeks)


if __name__ == "__main__":
    unittest.main()
