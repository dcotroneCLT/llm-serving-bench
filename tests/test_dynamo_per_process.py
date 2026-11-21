"""R5: opt-in Dynamo per-process split (monitoring/multiproc_monitor.py).

Default OFF must leave the historical sampler, CSV schema and file set exactly
as they are (the 19 completed Dynamo runs). ON writes perproc_<label> rows whose
per-tick sums equal the component sums, with a closed-set tag that separates the
Dynamo worker main process from the vLLM EngineCore inside a worker PGID.

Uses a REAL process tree: a fake worker (own session -> own PGID) spawning an
"EngineCore" child and an unrelated helper, sampled by the real monitor code.
Only the memory read is substituted (FAKE_MEMORY): USS/PSS need root on macOS
(task_for_pid) and on Linux need the sudo the production monitor runs under.
Process discovery, PGIDs, cmdlines, create_time and ppid are all real.

Run: python3 -m unittest tests.test_dynamo_per_process
"""
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "monitoring"))
sys.path.insert(0, str(REPO / "analysis"))

import multiproc_monitor as mpm  # noqa: E402
import validate_extension_run as ver  # noqa: E402
import psutil  # noqa: E402

# Deterministic stand-in for memory_full_info, derived from the (unprivileged)
# memory_info of the real process. Shared by the in-process tests and the
# monitor subprocess (see run_monitor).
FAKE_MEMORY = r"""
import collections, psutil
_Full = collections.namedtuple("_Full", "rss vms uss pss")
def _fake_full(self):
    m = psutil.Process.memory_info(self)
    return _Full(m.rss, m.vms, m.rss // 2, (m.rss * 3) // 4)
psutil.Process.memory_full_info = _fake_full
"""


def setUpModule():
    global _REAL_FULL
    _REAL_FULL = psutil.Process.memory_full_info
    exec(FAKE_MEMORY, {})


def tearDownModule():
    psutil.Process.memory_full_info = _REAL_FULL

WORKER = r"""
import subprocess, sys, time
kids = [
    # Built at runtime so this source text never contains the title itself.
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "VLLM::Engine" + "Core"]),
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "resource_tracker"]),
]
print("ready", flush=True)
time.sleep(120)
"""


class FakeWorker:
    """python3 -c WORKER fake_dynamo_worker --disaggregation-mode prefill, in its
    own session (PGID == pid), with two children inheriting the PGID."""

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-c", WORKER, "fake_dynamo_worker",
             "--disaggregation-mode", "prefill"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        assert self.proc.stdout.readline().strip() == "ready"
        # Give the children time to exec (their cmdline is the final argv).
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=10)
        self.proc.stdout.close()

    def spec(self, per_process):
        spec = {
            "engine_group": "dynamo",
            "root_pid": None,
            "components": [{
                "label": "dynamo_prefill", "pattern": "fake_dynamo_worker",
                "require": "--disaggregation-mode prefill", "group": "engine",
                "expected_count": 1, "pgids": [self.proc.pid],
            }],
        }
        if per_process is not None:
            spec["per_process_split"] = per_process
        return spec


class TickLevel(unittest.TestCase):
    def test_tags_and_sums_on_a_real_process_group(self):
        with FakeWorker() as w:
            comps = [mpm.Component(c) for c in w.spec(True)["components"]]
            rows, groups = mpm.tick(comps, None, time.time(), per_process=True)
            row = rows[0]
            self.assertTrue(row["membership_complete"])
            procs = row["_per_process"]
            self.assertEqual(len(procs), 3)
            self.assertEqual(sorted(p["tag"] for p in procs),
                             ["dynamo_main", "other", "vllm_enginecore"])
            self.assertTrue(all(p["tag"] in mpm.PROCESS_TAGS for p in procs))
            main = next(p for p in procs if p["tag"] == "dynamo_main")
            self.assertEqual(main["pid"], w.proc.pid)
            self.assertEqual({p["pid"] for p in procs},
                             {int(x) for x in row["pids"].split(",")})
            for field in ("uss_bytes", "pss_bytes", "rss_bytes"):
                self.assertEqual(sum(p[field] for p in procs), row[field], field)
            for p in procs:
                self.assertGreater(p["create_time"], 0)
                self.assertTrue(p["component_membership_complete"])

    def test_default_off_rows_have_the_historical_shape(self):
        with FakeWorker() as w:
            comps = [mpm.Component(c) for c in w.spec(None)["components"]]
            rows_off, groups_off = mpm.tick(comps, None, time.time())
            rows_on, groups_on = mpm.tick(comps, None, time.time(), per_process=True)
            self.assertNotIn("_per_process", rows_off[0])
            self.assertEqual(set(rows_on[0]) - {"_per_process"}, set(rows_off[0]))
            self.assertEqual(set(groups_on["engine"]), set(groups_off["engine"]))
            for key in ("n_pids_matched", "n_pids_sampled", "n_pgids_alive",
                        "membership_complete", "pids"):
                self.assertEqual(rows_on[0][key], rows_off[0][key], key)

    def test_infra_components_are_never_split(self):
        with FakeWorker() as w:
            spec = w.spec(True)
            spec["components"][0]["group"] = "infra"
            comps = [mpm.Component(c) for c in spec["components"]]
            rows, _ = mpm.tick(comps, None, time.time(), per_process=True)
            self.assertNotIn("_per_process", rows[0])

    def test_tag_classification_is_closed(self):
        comp = mpm.Component({"label": "x", "pattern": r"dynamo\.vllm"})
        self.assertEqual(mpm.process_tag("python3 -m dynamo.vllm --model q", comp), "dynamo_main")
        self.assertEqual(mpm.process_tag("VLLM::EngineCore", comp), "vllm_enginecore")
        self.assertEqual(mpm.process_tag("python3 -c from multiprocessing.resource_tracker", comp), "other")
        self.assertEqual(mpm.process_tag("", comp), "other")
        # A titled EngineCore whose cmdline kept the parent's argv is still found
        # through the (truncated) process name.
        self.assertEqual(mpm.process_tag("python3 -m dynamo.vllm", comp, "VLLM::EngineCor"),
                         "vllm_enginecore")
        self.assertEqual(mpm.process_tag("python3 -m dynamo.vllm", comp, "python3"),
                         "dynamo_main")
        self.assertEqual(mpm.process_tag("python3 -m dynamo.vllm", comp, "EngineCore_0"),
                         "vllm_enginecore")


def run_monitor(out_dir: Path, spec: dict, seconds: float = 3.0) -> None:
    comp_file = out_dir / "components.json"
    comp_file.write_text(json.dumps(spec))
    monitor = REPO / "monitoring" / "multiproc_monitor.py"
    launcher = (FAKE_MEMORY + "\nimport runpy, sys\n"
                f"sys.path.insert(0, {str(monitor.parent)!r})\n"
                f"sys.argv = [{str(monitor)!r}] + sys.argv[1:]\n"
                f"runpy.run_path({str(monitor)!r}, run_name='__main__')\n")
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher,
         "--components-file", str(comp_file), "--output-dir", str(out_dir),
         "--period-seconds", "0.5", "--rotation-seconds", "3600"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    time.sleep(seconds)
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=20)
    if proc.returncode not in (0, -signal.SIGTERM):
        raise AssertionError(f"monitor exited {proc.returncode}: {err}")


class MonitorProcess(unittest.TestCase):
    def test_flag_off_writes_exactly_the_historical_files(self):
        with FakeWorker() as w, tempfile.TemporaryDirectory() as t:
            out = Path(t)
            run_monitor(out, w.spec(None))
            names = sorted(p.name for p in out.glob("*.csv"))
            self.assertEqual(names, ["agg_dynamo_000000.csv", "dynamo_prefill_000000.csv"])
            with (out / "dynamo_prefill_000000.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), mpm.COMPONENT_FIELDS)
            with (out / "agg_dynamo_000000.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), mpm.AGG_FIELDS)

    def test_flag_false_is_the_same_as_absent(self):
        with FakeWorker() as w, tempfile.TemporaryDirectory() as t:
            out = Path(t)
            run_monitor(out, w.spec(False))
            self.assertEqual(list(out.glob("perproc_*.csv")), [])

    def test_flag_on_output_passes_the_validator_check(self):
        with FakeWorker() as w, tempfile.TemporaryDirectory() as t:
            out = Path(t)
            spec = w.spec(True)
            run_monitor(out, spec)
            with (out / "perproc_dynamo_prefill_000000.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), mpm.PERPROC_FIELDS)
            # Component and aggregate schemas are unchanged by the flag.
            with (out / "dynamo_prefill_000000.csv").open() as f:
                self.assertEqual(next(csv.reader(f)), mpm.COMPONENT_FIELDS)
            ok, msgs = ver.check_per_process_split(out, spec["components"])
            # One audit line per process identity, not per tick.
            idents = [json.loads(line) for line in
                      (out / "perproc_identities.jsonl").read_text().splitlines()]
            self.assertEqual(len(idents), 3)
            self.assertEqual(sorted(i["tag"] for i in idents),
                             ["dynamo_main", "other", "vllm_enginecore"])
            self.assertTrue(all(i["cmdline"] and i["name"] for i in idents))
            self.assertTrue(ok, msgs)
            self.assertTrue(any("vllm_enginecore" in m for m in msgs), msgs)
            # Per-process files never collide with the component/aggregate globs.
            comp_rows = ver._read_csvs(out, "dynamo_prefill")
            self.assertTrue(comp_rows)
            self.assertTrue(all("label" in r and r["label"] == "dynamo_prefill" and "tag" not in r
                                for r in comp_rows))


def _write(path: Path, fields, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class ValidatorCheck(unittest.TestCase):
    COMP = [{"label": "dynamo_decode", "group": "engine"}, {"label": "etcd", "group": "infra"}]

    def _layout(self, tmp: Path, perproc_rows, comp_uss=300):
        _write(tmp / "dynamo_decode_000000.csv", mpm.COMPONENT_FIELDS, [
            {"ts_unix": "1.0", "label": "dynamo_decode", "group": "engine",
             "membership_complete": "True", "pids": "10,11",
             "uss_bytes": comp_uss, "pss_bytes": 400, "rss_bytes": 500},
            {"ts_unix": "2.0", "label": "dynamo_decode", "group": "engine",
             "membership_complete": "False", "pids": "10", "uss_bytes": 1,
             "pss_bytes": 1, "rss_bytes": 1},
        ])
        _write(tmp / "perproc_dynamo_decode_000000.csv", mpm.PERPROC_FIELDS, perproc_rows)

    def _good_rows(self):
        return [
            {"ts_unix": "1.0", "label": "dynamo_decode", "pid": 10, "create_time": 5.0,
             "tag": "dynamo_main", "uss_bytes": 100, "pss_bytes": 150, "rss_bytes": 200},
            {"ts_unix": "1.0", "label": "dynamo_decode", "pid": 11, "create_time": 6.0,
             "tag": "vllm_enginecore", "uss_bytes": 200, "pss_bytes": 250, "rss_bytes": 300},
        ]

    def test_consistent_split_passes_and_infra_is_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            self._layout(Path(t), self._good_rows())
            ok, msgs = ver.check_per_process_split(Path(t), self.COMP)
            self.assertTrue(ok, msgs)

    def test_sum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            self._layout(Path(t), self._good_rows(), comp_uss=310)
            ok, _ = ver.check_per_process_split(Path(t), self.COMP)
            self.assertFalse(ok)

    def test_pid_set_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            rows = self._good_rows()
            rows[1]["pid"] = 12
            self._layout(Path(t), rows)
            ok, _ = ver.check_per_process_split(Path(t), self.COMP)
            self.assertFalse(ok)

    def test_unknown_tag_fails(self):
        with tempfile.TemporaryDirectory() as t:
            rows = self._good_rows()
            rows[1]["tag"] = "prune"
            self._layout(Path(t), rows)
            ok, _ = ver.check_per_process_split(Path(t), self.COMP)
            self.assertFalse(ok)

    def test_missing_rows_fail(self):
        with tempfile.TemporaryDirectory() as t:
            self._layout(Path(t), [])
            ok, _ = ver.check_per_process_split(Path(t), self.COMP)
            self.assertFalse(ok)
            (Path(t) / "perproc_dynamo_decode_000000.csv").unlink()
            ok, _ = ver.check_per_process_split(Path(t), self.COMP)
            self.assertFalse(ok)


class ExistingDynamoCellsUnchanged(unittest.TestCase):
    def test_no_existing_dynamo_cell_opts_in(self):
        cells = sorted((REPO / "campaigns" / "extension" / "cells").glob("**/*dynamo*.yaml"))
        self.assertGreaterEqual(len(cells), 21)
        for path in cells:
            if path.stem == "dow_dynamo_disagg_cp4":
                continue  # the R2 bridge center point opts in on purpose
            comps = yaml.safe_load(path.read_text())["monitors"]["components"]
            self.assertNotIn("per_process_split", comps, path.name)


if __name__ == "__main__":
    unittest.main()
