"""The DoW design is the science -- test it like code.

Covers scripts/generate_dow_cells.py and the artifacts it emits under
campaigns/extension:
  * matrix correctness (Res V I=ABCDE, balance, center codes);
  * generation determinism + idempotency (byte-identical trees; seed changes
    the order but not the set);
  * schedule properties (all 57 once, interleaving, 48h only for Dynamo CPs,
    center-point early/middle/late spread per system);
  * yaml integrity (parses, engine/monitors byte-identical to the template,
    per-cell required calibration, physical values match the coded matrix);
  * the campaign loads through the real campaign.py loader: serial, 57 runs,
    its OWN state file distinct from the validation campaign.

Run: python3 -m unittest tests.test_dow_campaign
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import generate_dow_cells as gen  # noqa: E402
import campaign as camp  # noqa: E402

import yaml  # noqa: E402

EXT = REPO / "campaigns" / "extension"
CELLS_DIR = EXT / "cells" / "dow"


# --------------------------------------------------------------------------
# 1. Design matrix correctness
# --------------------------------------------------------------------------


class DesignMatrix(unittest.TestCase):
    def setUp(self):
        self.rows = gen.coded_design()

    def test_sixteen_unique_coded_rows(self):
        self.assertEqual(len(self.rows), 16)
        uniq = {(r["A"], r["B"], r["C"], r["D"], r["E"]) for r in self.rows}
        self.assertEqual(len(uniq), 16)

    def test_each_factor_balanced_8_8(self):
        for f in ("A", "B", "C", "D", "E"):
            plus = sum(1 for r in self.rows if r[f] == 1)
            minus = sum(1 for r in self.rows if r[f] == -1)
            self.assertEqual((plus, minus), (8, 8), f"{f} not balanced")

    def test_defining_relation_E_equals_ABCD(self):
        # I=ABCDE  <=>  E = A*B*C*D for every row.
        for r in self.rows:
            self.assertEqual(r["E"], r["A"] * r["B"] * r["C"] * r["D"], r)

    def test_all_two_factor_interaction_columns_balanced(self):
        # Resolution V => every 2-factor interaction column is also balanced
        # (a further check that the fraction is the intended one).
        facs = ("A", "B", "C", "D", "E")
        for i in range(len(facs)):
            for j in range(i + 1, len(facs)):
                col = [r[facs[i]] * r[facs[j]] for r in self.rows]
                self.assertEqual(sum(col), 0, f"{facs[i]}{facs[j]} not balanced")

    def test_center_codes_are_zero(self):
        self.assertEqual(gen.CENTER_CODES, {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0})

    def test_physical_center_is_midpoint(self):
        phys = gen.physical(dict(gen.CENTER_CODES, yates=None))
        self.assertEqual(phys["rate_fraction"], 0.575)
        self.assertEqual(phys["prompt_len"], 3256)
        self.assertEqual(phys["max_tokens"], 544)
        self.assertEqual(phys["prefix_repeat_fraction"], 0.4)
        self.assertEqual(phys["arrival_mode"], "bursty")
        self.assertEqual(phys["burst_factor"], 2.5)


# --------------------------------------------------------------------------
# 2. Generation determinism + idempotency
# --------------------------------------------------------------------------


def _tree_snapshot(base: Path) -> dict:
    out = {}
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(base))] = p.read_text()
    return out


class GenerationDeterminism(unittest.TestCase):
    def test_two_runs_byte_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            gen.generate(Path(a))
            gen.generate(Path(b))
            self.assertEqual(_tree_snapshot(Path(a)), _tree_snapshot(Path(b)))

    def test_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as a:
            base = Path(a)
            gen.generate(base)
            second = gen.generate(base)  # already up to date
            self.assertEqual(second["changed"], [])
            self.assertEqual(second["removed"], [])

    def test_check_mode_reports_up_to_date_after_generate(self):
        with tempfile.TemporaryDirectory() as a:
            base = Path(a)
            gen.generate(base)
            res = gen.generate(base, check=True)
            self.assertEqual(res["changed"], [])
            self.assertEqual(res["removed"], [])

    def test_stale_generated_cell_is_removed(self):
        with tempfile.TemporaryDirectory() as a:
            base = Path(a)
            gen.generate(base)
            stray = base / "cells" / "dow" / "dow_dynamo_disagg_p99.yaml"
            stray.write_text("stale\n")
            res = gen.generate(base)
            self.assertIn(str(stray), res["removed"])
            self.assertFalse(stray.exists())

    def test_seed_changes_order_not_set(self):
        o1 = gen.schedule_order(gen.DESIGN_SEED)
        o2 = gen.schedule_order(gen.DESIGN_SEED + 1)
        self.assertNotEqual(o1, o2)                       # order differs
        self.assertEqual({c for _, c in o1}, {c for _, c in o2})  # same set
        self.assertEqual(len({c for _, c in o1}), 57)


# --------------------------------------------------------------------------
# 3. Schedule properties
# --------------------------------------------------------------------------


class Schedule(unittest.TestCase):
    def setUp(self):
        self.order = gen.schedule_order()

    def test_all_57_present_exactly_once(self):
        ids = [c for _, c in self.order]
        self.assertEqual(len(ids), 57)
        self.assertEqual(len(set(ids)), 57)
        # 19 per system.
        for system in gen.SYSTEMS:
            self.assertEqual(sum(1 for s, _ in self.order if s == system), 19)

    def test_no_more_than_two_consecutive_same_system(self):
        systems = [s for s, _ in self.order]
        run = 1
        worst = 1
        for i in range(1, len(systems)):
            run = run + 1 if systems[i] == systems[i - 1] else 1
            worst = max(worst, run)
        self.assertLessEqual(worst, 2, "a system is concentrated (>2 consecutive)")

    def test_center_points_spread_early_middle_late_per_system(self):
        # Within each system's subsequence (its 19 positions in schedule order),
        # the three CPs land one in each third.
        for system in gen.SYSTEMS:
            positions = [i for i, (s, c) in enumerate(self.order)
                         if s == system and "_cp" in c]
            # 0-based index within THIS system's 19-run subsequence.
            sub_idx = []
            k = 0
            for s, c in self.order:
                if s == system:
                    if "_cp" in c:
                        sub_idx.append(k)
                    k += 1
            self.assertEqual(len(sub_idx), 3, system)
            thirds = {i // 7 for i in sub_idx}  # 19/3 ~ 6.33 -> thirds 0,1,2 via //7 (0-6,7-13,14-18)
            # Use explicit third boundaries to avoid off-by-one in the //7 shortcut.
            first = sum(1 for i in sub_idx if i < 7)
            mid = sum(1 for i in sub_idx if 7 <= i < 13)
            late = sum(1 for i in sub_idx if i >= 13)
            self.assertEqual((first, mid, late), (1, 1, 1), f"{system}: {sub_idx}")


# --------------------------------------------------------------------------
# 4. Yaml integrity of the checked-in generated tree
# --------------------------------------------------------------------------


def _load_matrix_csv() -> list[dict]:
    text = (EXT / "dow_design_matrix.csv").read_text()
    data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(data_lines))


class YamlIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _load_matrix_csv()
        cls.templates = {s: yaml.safe_load(gen.TEMPLATE_CELL[s].read_text())
                         for s in gen.SYSTEMS}

    def test_all_57_cells_exist_and_parse(self):
        self.assertEqual(len(self.rows), 57)
        for row in self.rows:
            path = CELLS_DIR / f"{row['cell_id']}.yaml"
            self.assertTrue(path.exists(), path)
            doc = yaml.safe_load(path.read_text())
            self.assertEqual(doc["cell_id"], row["cell_id"])

    def test_engine_and_monitors_byte_identical_to_template(self):
        for row in self.rows:
            doc = yaml.safe_load((CELLS_DIR / f"{row['cell_id']}.yaml").read_text())
            tmpl = self.templates[row["system"]]
            # Structural equality is the invariant; also assert the canonical
            # serialization matches (byte-identical section).
            self.assertEqual(doc["engine"], tmpl["engine"], row["cell_id"])
            self.assertEqual(doc["monitors"], tmpl["monitors"], row["cell_id"])
            self.assertEqual(
                yaml.safe_dump(doc["engine"], sort_keys=False),
                yaml.safe_dump(tmpl["engine"], sort_keys=False), row["cell_id"])
            self.assertEqual(
                yaml.safe_dump(doc["monitors"], sort_keys=False),
                yaml.safe_dump(tmpl["monitors"], sort_keys=False), row["cell_id"])

    def test_physical_values_match_the_coded_matrix(self):
        for row in self.rows:
            doc = yaml.safe_load((CELLS_DIR / f"{row['cell_id']}.yaml").read_text())
            over = doc["workload"]["client_config_overrides"]
            # Length factors: deterministic point level == matrix physical value.
            self.assertEqual(over["prompt_len"]["median"], int(row["prompt_len"]), row["cell_id"])
            self.assertEqual(over["prompt_len"]["min"], int(row["prompt_len"]))
            self.assertEqual(over["prompt_len"]["max"], int(row["prompt_len"]))
            self.assertEqual(over["max_tokens"]["median"], int(row["max_tokens"]))
            self.assertAlmostEqual(over["prefix_repeat_fraction"],
                                   float(row["prefix_repeat_fraction"]))
            self.assertEqual(over["shared_prefix_len"], int(row["shared_prefix_len"]))
            self.assertEqual(over["arrival_mode"], row["arrival_mode"])
            # Rate factor lives in the calibration metadata, not target_rate_rps.
            self.assertNotIn("target_rate_rps", over, f"{row['cell_id']} must not fix a rate")
            self.assertAlmostEqual(doc["calibration_fraction"], float(row["rate_fraction"]))
            # Burstiness encoding convention.
            if row["arrival_mode"] == "bursty":
                self.assertAlmostEqual(over["burst_factor"], float(row["burst_factor"]))
                self.assertAlmostEqual(over["burst_on_seconds"], float(row["burst_on_seconds"]))
            else:
                self.assertEqual(row["arrival_mode"], "poisson")
                self.assertNotIn("burst_factor", over)

    def test_shared_prefix_512_iff_fraction_positive(self):
        for row in self.rows:
            frac = float(row["prefix_repeat_fraction"])
            spl = int(row["shared_prefix_len"])
            self.assertEqual(spl, 512 if frac > 0 else 0, row["cell_id"])

    def test_windows_only_dynamo_center_points_are_48h(self):
        forty_eight = []
        for row in self.rows:
            doc = yaml.safe_load((CELLS_DIR / f"{row['cell_id']}.yaml").read_text())
            self.assertEqual(doc["warmup_discard_s"], 3600, row["cell_id"])
            if doc["duration_s"] == gen.DUR_48H:
                forty_eight.append(row["cell_id"])
            else:
                self.assertEqual(doc["duration_s"], gen.DUR_36H, row["cell_id"])
        self.assertEqual(sorted(forty_eight),
                         ["dow_dynamo_disagg_cp1", "dow_dynamo_disagg_cp2",
                          "dow_dynamo_disagg_cp3"])


# --------------------------------------------------------------------------
# 5. The descriptor loads through the real campaign.py loader
# --------------------------------------------------------------------------


class CampaignLoads(unittest.TestCase):
    def setUp(self):
        self.yaml_path = EXT / "dow_campaign.yaml"
        self.campaign = camp.load_campaign(self.yaml_path)

    def test_serial_with_own_state_file(self):
        self.assertEqual(self.campaign["mode"], "serial")
        self.assertEqual(self.campaign["campaign_id"], "extension_dow_screening")
        self.assertEqual(self.campaign.get("state_file"), "state/dow_campaign_state.json")
        # Distinct from BOTH the validation campaign and the long test.
        val = camp.load_campaign(EXT / "campaign.yaml")
        self.assertNotEqual(self.campaign.get("state_file"), val.get("state_file"))

    def test_schedule_is_57_in_generated_order(self):
        sched = camp.build_schedule(self.campaign, self.yaml_path)
        self.assertEqual(len(sched), 57)
        # replicas_per_cell:1 => cells list order == execution order == generator order.
        gen_order = [c for _, c in gen.schedule_order()]
        self.assertEqual([s.cell_id for s in sched], gen_order)

    def test_every_run_requires_a_distinct_calibration_file(self):
        sched = camp.build_schedule(self.campaign, self.yaml_path)
        files = []
        for s in sched:
            self.assertTrue(s.calibration_required, s.cell_id)
            self.assertIsNotNone(s.calibration_file, s.cell_id)
            self.assertTrue(s.calibration_file.endswith(f"{s.cell_id}.json"), s.cell_id)
            files.append(s.calibration_file)
        self.assertEqual(len(files), len(set(files)), "calibration files must be per-cell")

    def test_disk_floors_are_explicit_for_unattended_runs(self):
        # A 57-run unattended campaign must not rely on the code default inode
        # floor; the descriptor declares all three disk floors explicitly and
        # keeps the inode floor in sync with the validation campaign.
        val = camp.load_campaign(EXT / "campaign.yaml")
        self.assertEqual(self.campaign["min_free_gb"], 50)
        self.assertEqual(self.campaign["min_free_gb_mid_run"], 25)
        self.assertIn("min_inodes_free_mid_run", self.campaign)
        self.assertEqual(self.campaign["min_inodes_free_mid_run"],
                         val["min_inodes_free_mid_run"])

    def test_calibration_policy_is_generated_not_hand_edited(self):
        # The v1 finite-window ceilings are REFUSED (hard method floor), and the
        # 13-week serial schedule needs a max-age wide enough that a wk1
        # calibration is still fresh for the tail run (~85 days out) -- see the
        # arithmetic in the 2026-07-10 amendment. Both knobs MUST be emitted by
        # the generator: a value hand-added to this DO-NOT-EDIT file would be
        # silently stripped by the next regeneration (the exact class of bug that
        # once let a leftover v1 ceiling downgrade to a warning).
        self.assertEqual(self.campaign.get("calibration_min_method_version"), 2)
        self.assertGreaterEqual(self.campaign.get("calibration_max_age_days", 0), 90)

    def test_dynamo_center_points_are_the_only_48h_runs_in_schedule(self):
        sched = camp.build_schedule(self.campaign, self.yaml_path)
        long_runs = [s.cell_id for s in sched if s.duration_s == gen.DUR_48H]
        self.assertEqual(sorted(long_runs),
                         ["dow_dynamo_disagg_cp1", "dow_dynamo_disagg_cp2",
                          "dow_dynamo_disagg_cp3"])


if __name__ == "__main__":
    unittest.main()
