"""R2: the Triton/vLLM DoW rerun as three <= 25-day machine windows.

Checks the GENERATED descriptors on disk (not only the generator's in-memory
plan): window run counts, quarter-block blocking (mains balanced, only AC/AD/CD
exposed), center-point placement, interleave, projected durations with the
retry margin, per-window state and calibration paths, the reused container_tree
v2 cells, the new Dynamo bridge cell, and generator --check / determinism.

Run: python3 -m unittest tests.test_rerun_windows
"""
import csv
import io
import itertools
import sys
import tempfile
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "campaigns" / "extension"
sys.path.insert(0, str(REPO / "scripts"))

import campaign as camp  # noqa: E402
import generate_dow_cells as gdc  # noqa: E402
import generate_rerun_windows as grw  # noqa: E402

FACTORS = ("A", "B", "C", "D", "E")
TWO_FI = ["".join(p) for p in itertools.combinations(FACTORS, 2)]


def descriptor(k):
    return yaml.safe_load((EXT / f"dow_rerun_w{k}.yaml").read_text())


def cell_doc(entry):
    return yaml.safe_load((EXT / entry["yaml"]).read_text())


def codes(cid):
    return grw._codes_of(cid)


def system_of(cid):
    return "dynamo_disagg" if cid.startswith("dow_dynamo") else cid.split("_")[1]


class GeneratedTreeIsCurrent(unittest.TestCase):
    def test_generator_check_clean(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(grw.main(["--check"]), 0)
            self.assertEqual(gdc.main(["--check"]), 0)

    def test_generator_is_idempotent_and_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "extension"
            shutil.copytree(EXT, out, ignore=shutil.ignore_patterns("state"))
            for name in ("dow_rerun_w1.yaml", "dow_rerun_w2.yaml", "dow_rerun_w3.yaml",
                         "dow_rerun_w1_soak.yaml", "rerun_design_matrix.csv"):
                (out / name).unlink()
            shutil.rmtree(out / "cells" / "rerun")
            # gdc reads its templates from the real tree; the rerun generator
            # writes into the copy. Generated text must match the committed tree.
            first = grw.generate(out)
            self.assertEqual(len(first["changed"]), 8)
            second = grw.generate(out)
            self.assertEqual(second["changed"], [])
            for name in ("dow_rerun_w1.yaml", "dow_rerun_w2.yaml", "dow_rerun_w3.yaml",
                         "dow_rerun_w1_soak.yaml", "rerun_design_matrix.csv"):
                self.assertEqual((out / name).read_text(), (EXT / name).read_text(), name)

    def test_seed_changes_order_not_allocation(self):
        a, b = grw.window_order(grw.RERUN_SEED), grw.window_order(grw.RERUN_SEED + 1)
        self.assertNotEqual(a, b)
        for k in grw.WINDOWS:
            self.assertEqual(sorted(a[k]), sorted(b[k]))
        grw.check_plan(b)  # every seed yields a valid plan

    def test_original_campaign_untouched(self):
        # The closed 57-run descriptor and the 19 Dynamo cells are not rewritten.
        self.assertEqual(gdc.generate(EXT, check=True)["changed"], [])
        self.assertFalse((EXT / "cells" / "dow" / "dow_dynamo_disagg_cp4.yaml").exists())


class WindowContent(unittest.TestCase):
    def test_run_counts_14_14_11(self):
        self.assertEqual([len(descriptor(k)["cells"]) for k in (1, 2, 3)], [14, 14, 11])

    def test_every_rerun_point_exactly_once(self):
        all_ids = [c["id"] for k in (1, 2, 3) for c in descriptor(k)["cells"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        expected = {gdc.cell_id(s, "p", n) for s in ("triton", "vllm") for n in range(1, 17)}
        expected |= {gdc.cell_id(s, "cp", k) for s in ("triton", "vllm") for k in (1, 2, 3)}
        expected.add("dow_dynamo_disagg_cp4")
        self.assertEqual(set(all_ids), expected)

    def test_design_split_8_4_4_and_4_8_4(self):
        for system, split in (("triton", [8, 4, 4]), ("vllm", [4, 8, 4])):
            got = [sum(1 for c in descriptor(k)["cells"]
                       if system_of(c["id"]) == system and "_p" in c["id"]) for k in (1, 2, 3)]
            self.assertEqual(got, split, system)

    def test_mains_balanced_per_window_per_system(self):
        for k in (1, 2, 3):
            for system in ("triton", "vllm"):
                rows = [codes(c["id"]) for c in descriptor(k)["cells"]
                        if system_of(c["id"]) == system and "_p" in c["id"]]
                for f in FACTORS:
                    self.assertEqual(sum(r[f] for r in rows), 0, (k, system, f))

    def test_window_exposure_limited_to_ac_ad_cd(self):
        """Only AC, AD, CD may be correlated with window membership, and the
        windows are indeed made of whole quarter blocks (AC, AD constant)."""
        for system in ("triton", "vllm"):
            for k in (1, 2, 3):
                rows = [codes(c["id"]) for c in descriptor(k)["cells"]
                        if system_of(c["id"]) == system and "_p" in c["id"]]
                for name in TWO_FI:
                    total = sum(grw.interaction(r, name) for r in rows)
                    if name not in ("AC", "AD", "CD"):
                        self.assertEqual(total, 0, (system, k, name))
                quarters = {grw.quarter_of(r) for r in rows}
                self.assertEqual(len(rows), 4 * len(quarters))
                for q in quarters:
                    self.assertEqual(sum(1 for r in rows if grw.quarter_of(r) == q), 4)

    def test_one_center_point_per_system_near_the_middle(self):
        for k in (1, 2, 3):
            ids = [c["id"] for c in descriptor(k)["cells"]]
            n = len(ids)
            for system in ("triton", "vllm"):
                cps = [i for i, cid in enumerate(ids) if system_of(cid) == system and "_cp" in cid]
                self.assertEqual([ids[i] for i in cps], [gdc.cell_id(system, "cp", k)])
                self.assertLessEqual(abs(cps[0] - (n - 1) / 2), 1.0, (k, system))
        w3 = [c["id"] for c in descriptor(3)["cells"]]
        self.assertEqual(w3.index("dow_dynamo_disagg_cp4"), (len(w3) - 1) // 2)
        for k in (1, 2):
            self.assertNotIn("dow_dynamo_disagg_cp4", [c["id"] for c in descriptor(k)["cells"]])

    def test_systems_interleaved_as_evenly_as_the_split_allows(self):
        for k in (1, 2, 3):
            seq = [system_of(c["id"]) for c in descriptor(k)["cells"]]
            runs = [len(list(g)) for _, g in itertools.groupby(seq)]
            self.assertLessEqual(max(runs), 2, (k, seq))
            design = [system_of(c["id"]) for c in descriptor(k)["cells"] if "_p" in c["id"]]
            counts = {s: design.count(s) for s in ("triton", "vllm")}
            minority = min(counts, key=counts.get)
            pos = [i for i, s in enumerate(design) if s == minority]
            gaps = [b - a for a, b in zip(pos, pos[1:])]
            self.assertLessEqual(max(gaps) - min(gaps), 1, (k, design))

    def test_order_matches_design_matrix_csv(self):
        text = (EXT / "rerun_design_matrix.csv").read_text()
        rows = list(csv.DictReader(line for line in io.StringIO(text) if not line.startswith("#")))
        self.assertEqual(len(rows), 39)
        for k in (1, 2, 3):
            mine = sorted((r for r in rows if r["window"] == str(k)), key=lambda r: int(r["position"]))
            self.assertEqual([r["cell_id"] for r in mine], [c["id"] for c in descriptor(k)["cells"]])
            for r in mine:
                c = codes(r["cell_id"])
                self.assertEqual([int(r[f"{f}_code"]) for f in FACTORS], [c[f] for f in FACTORS])
                phys = gdc.physical(c)
                self.assertEqual(int(r["prompt_len"]), phys["prompt_len"])
                self.assertEqual(int(r["max_tokens"]), phys["max_tokens"])
                self.assertEqual(r["arrival_mode"], phys["arrival_mode"])
                self.assertEqual(r["calibration_file"],
                                 f"state/calibration_rerun/w{k}/{r['cell_id']}.json")


class Durations(unittest.TestCase):
    def test_projection_within_25_days_with_retry_margin(self):
        for k in (1, 2, 3):
            cells = descriptor(k)["cells"]
            runs_h = sum(cell_doc(c)["duration_s"] for c in cells) / 3600
            calib_h = len(cells) * 17.0 / 14.0
            soak_h = 2 * 6.0 if k == 1 else 0.0
            n_soak = 2 if k == 1 else 0
            overhead_h = (len(cells) + n_soak) * grw.RUN_OVERHEAD_H + grw.WINDOW_SETUP_H
            total_d = (runs_h + calib_h + soak_h + overhead_h) / 24
            margin_d = 25.0 - total_d
            self.assertGreaterEqual(margin_d, 1.5, k)
            longest_d = max(cell_doc(c)["duration_s"] for c in cells) / 86400
            self.assertGreaterEqual(margin_d, longest_d, k)  # one retry of the longest run fits
            meta = descriptor(k)["rerun_window"]
            self.assertAlmostEqual(meta["projected_total_days"], total_d, places=2)
            self.assertAlmostEqual(meta["projected_retry_margin_days"], margin_d, places=2)
        # The calibration pace is the stated 17 h for 14 cells.
        self.assertAlmostEqual(descriptor(1)["rerun_window"]["projected_calibration_h"], 17.0)
        self.assertEqual(descriptor(1)["rerun_window"]["projected_soak_h"], 12.0)
        self.assertEqual(descriptor(2)["rerun_window"]["projected_soak_h"], 0.0)

    def test_infeasible_plan_is_refused(self):
        orders = grw.window_order()
        orders[1] = orders[1] + [("triton", "dow_triton_p03")] * 2
        with self.assertRaises(grw.PlanError):
            grw.check_plan(orders)


class DescriptorsLoadAndIsolate(unittest.TestCase):
    def test_descriptors_load_with_own_state_and_calibration_dirs(self):
        states = set()
        for k in (1, 2, 3):
            path = EXT / f"dow_rerun_w{k}.yaml"
            d = camp.load_campaign(path)
            self.assertEqual(d["campaign_id"], "extension_dow_rescoped")
            self.assertEqual(d["calibration_min_method_version"], 2)
            self.assertEqual(d["calibration_max_age_days"], 30)
            self.assertEqual(d["rerun_window"]["calibration_dir"], f"state/calibration_rerun/w{k}")
            states.add(d["state_file"])
            schedule = camp.build_schedule(d, path)
            self.assertEqual(len(schedule), len(d["cells"]))
            for spec in schedule:
                self.assertTrue(spec.calibration_required)
                self.assertEqual(Path(spec.calibration_file),
                                 (EXT / f"state/calibration_rerun/w{k}/{spec.cell_id}.json").resolve())
                self.assertIsNotNone(spec.calibration_fraction)
        soak = camp.load_campaign(EXT / "dow_rerun_w1_soak.yaml")
        states.add(soak["state_file"])
        states.add(yaml.safe_load((EXT / "dow_campaign.yaml").read_text())["state_file"])
        states.add(yaml.safe_load((EXT / "dow_dynamo_tail.yaml").read_text())["state_file"])
        self.assertEqual(len(states), 6)

    def test_rerun_cells_are_the_existing_container_tree_v2_cells(self):
        for k in (1, 2, 3):
            for c in descriptor(k)["cells"]:
                if c["id"] == "dow_dynamo_disagg_cp4":
                    continue
                self.assertEqual(c["yaml"], f"cells/dow/{c['id']}.yaml")
                comps = cell_doc(c)["monitors"]["components"]
                self.assertEqual((comps["scope_type"], comps["scope_version"]), ("container_tree", 2))


class BridgeCell(unittest.TestCase):
    def test_cp4_is_cp1_plus_per_process_split(self):
        cp1 = yaml.safe_load((EXT / "cells" / "dow" / "dow_dynamo_disagg_cp1.yaml").read_text())
        cp4 = yaml.safe_load((EXT / "cells" / "rerun" / "dow_dynamo_disagg_cp4.yaml").read_text())
        self.assertEqual(cp4["cell_id"], "dow_dynamo_disagg_cp4")
        self.assertEqual(cp4["engine"], cp1["engine"])
        self.assertEqual(cp4["workload"], cp1["workload"])
        self.assertEqual(cp4["duration_s"], 172_800)
        self.assertEqual(cp4["warmup_discard_s"], cp1["warmup_discard_s"])
        self.assertEqual(cp4["calibration_fraction"], cp1["calibration_fraction"])
        monitors = cp4["monitors"]
        self.assertIs(monitors["components"].pop("per_process_split"), True)
        self.assertEqual(monitors, cp1["monitors"])


if __name__ == "__main__":
    unittest.main()
