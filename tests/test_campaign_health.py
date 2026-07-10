"""End-to-end coverage for scripts/campaign_health.sh Section B.

The Section B per-run logic is lifecycle-aware and safety-critical (it is the
operator's blindfold-off view of a live 48h DoW run), but it is a bash script,
so it is exercised here by DRIVING THE REAL SCRIPT over synthetic run dirs and
asserting on the findings it emits -- the scenarios previously only smoke-tested
by hand. No docker / GPU: Section A degrades on this host (docker/nvidia-smi
absent) and we assert only on B.<run> findings.

Also covers the writer/reader path contract: log_mitigation.sh must write the
mitigations log where campaign_health.sh reads it (both derived from the
campaign yaml).

Run: python3 -m unittest tests.test_campaign_health
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEALTH = REPO / "scripts" / "campaign_health.sh"
LOG_MIT = REPO / "scripts" / "log_mitigation.sh"

DIGEST = "sha256:abc123"
START = 1_700_000_000
END = START + 172_800  # 48h -> ended (IS_RUNNING=0, so docker/GPU liveness skipped)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FINDING = re.compile(r"^\[(PASS|WARN|FAIL)\] (\S+) \| (.*)$")


def _have_bash() -> bool:
    return shutil.which("bash") is not None


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _gpu_csv(rd: Path, gi: int, n: int = 10) -> None:
    rows = "".join(f"{START+i},{gi},{40_000_000_000},0,0\n" for i in range(n))
    _write(rd / f"gpu{gi}_000000.csv",
           "ts_unix,gpu_index,vram_used_bytes,ecc_db_volatile,ecc_sb_volatile\n" + rows)


def _system_csv(rd: Path, n: int = 10) -> None:
    rows = "".join(f"{START+i*5},20,{8_000_000_000},0\n" for i in range(n))
    _write(rd / "system_000000.csv",
           "ts_unix,cpu_percent,mem_used_bytes,swap_used_bytes\n" + rows)


def _client_csv(rd: Path, n: int = 40) -> None:
    rows = "".join(f"ok,{START+i},{START+i+1}\n" for i in range(n))
    _write(rd / "client" / "requests_000000.csv",
           "status,submitted_at_unix,finished_at_unix\n" + rows)


def _agg_csv(rd: Path, unexpected: int = 0, with_col: bool = True, n: int = 10) -> None:
    cols = ["ts_unix", "group", "process_alive", "membership_complete",
            "n_components_expected", "n_components_complete", "n_pids_sampled",
            "n_pids_unexpected", "uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes",
            "_sample_duration_s", "_wall_clock_unix"]
    if not with_col:
        cols.remove("n_pids_unexpected")
    lines = [",".join(cols)]
    for i in range(n):
        ts = START + i * 5
        row = {"ts_unix": ts, "group": "engine", "process_alive": "True",
               "membership_complete": "True", "n_components_expected": 3,
               "n_components_complete": 3, "n_pids_sampled": 3,
               "n_pids_unexpected": unexpected, "uss_bytes": 2_000_000_000,
               "rss_bytes": 3_000_000_000, "vms_bytes": 9_000_000_000,
               "pss_bytes": 2_500_000_000, "_sample_duration_s": 0.1,
               "_wall_clock_unix": ts}
        lines.append(",".join(str(row[c]) for c in cols))
    _write(rd / "agg_engine_000000.csv", "\n".join(lines) + "\n")


def _proc_csv(rd: Path, label: str, n: int = 10) -> None:
    # Single-container proc monitor CSV: ts_unix, process_alive, rss_bytes.
    rows = "".join(f"{START+i*5},True,{2_000_000_000}\n" for i in range(n))
    _write(rd / f"{label}_000000.csv", "ts_unix,process_alive,rss_bytes\n" + rows)


def _dyn_common() -> dict:
    return dict(replica=1, started_at_unix=START, ended_at="x", ended_at_unix=END,
                image={"digest": DIGEST}, proc_prefix="agg_engine", duration_s=172_800,
                workload={"client_config_overrides": {"model": "M"}},
                interrupted_early=False, client_forced_kill=False,
                client_summary={"total": 40, "ok": 40})


_IDENTITY = {
    "dynamo_prefill": {"containers": ["dyn_prefill_1"], "pgids": [123], "expected_count": 1},
    "dynamo_decode": {"containers": ["dyn_decode_1"], "pgids": [124], "expected_count": 1},
}


def build_campaign(tmp: Path, campaign_id: str = "exthc") -> tuple[Path, Path]:
    cdir = tmp / "camp"
    (cdir / "state").mkdir(parents=True)
    runs = tmp / "runs"
    runs.mkdir()
    _write(cdir / "campaign.yaml",
           f"campaign_id: {campaign_id}\nmode: serial\nruns_root: {runs}\n"
           f"state_file: state/campaign_state.json\ncells: []\n")
    return cdir / "campaign.yaml", runs


def add_single_container(runs: Path, cid: str, name: str) -> Path:
    rd = runs / f"{cid}_{name}_r01"
    (rd / "logs").mkdir(parents=True)
    manifest = {
        "cell_id": name, "replica": 1,
        "engine": {"gpu_device": 0, "pid_strategy": {"type": "container_pid1"}},
        "container": {"name": f"{name}_r01", "host_pid": 4321},
        "monitors": {"proc": {"label": "eng"}},
        "image": {"digest": DIGEST},
        "started_at_unix": START, "ended_at": "x", "ended_at_unix": END,
        "duration_s": 172_800,
        "workload": {"client_config_overrides": {"model": "M"}},
        "interrupted_early": False, "client_forced_kill": False,
        "client_summary": {"total": 40, "ok": 40},
    }
    _write(rd / "manifest.json", json.dumps(manifest))
    _write(rd / "image_digest.txt", DIGEST + "\n")
    _write(rd / "docker_inspect.json", "{}\n")
    _write(rd / "logs" / "docker.log", "ok\n")
    _write(rd / "engine.pid", "4321\n")
    _proc_csv(rd, "eng")
    _gpu_csv(rd, 0)
    _system_csv(rd)
    _client_csv(rd)
    return rd


def add_dynamo(runs: Path, cid: str, name: str, *, shape: str = "launch",
               unexpected: int = 0, with_col: bool = True,
               gpus: tuple[int, int] = (0, 1)) -> Path:
    rd = runs / f"{cid}_{name}_r01"
    (rd / "logs").mkdir(parents=True)
    m = _dyn_common()
    m["cell_id"] = name
    if shape == "launch":
        # launch_cell shape: top-level lifecycle/topology/components.
        m.update(lifecycle="dynamo_disagg",
                 topology={"prefill_gpu": gpus[0], "decode_gpu": gpus[1]},
                 gpu_devices=list(gpus), components=_IDENTITY,
                 engine={"lifecycle": "dynamo_disagg",
                         "topology": {"prefill_gpu": gpus[0], "decode_gpu": gpus[1]}})
    elif shape == "attach":
        # attach_run shape: identity under monitors.components.components, lifecycle
        # /topology ONLY under engine.* (JSON), NO top-level keys, NO cell yaml.
        m.update(attach=True,
                 engine={"lifecycle": "dynamo_disagg",
                         "topology": {"prefill_gpu": gpus[0], "decode_gpu": gpus[1]}},
                 monitors={"components": {"engine_group": "engine", "components": [
                     {"label": "dynamo_prefill", "pgids": [123]},
                     {"label": "dynamo_decode", "pgids": [124]}]}})
    else:
        raise ValueError(shape)
    _write(rd / "manifest.json", json.dumps(m))
    _write(rd / "image_digest.txt", DIGEST + "\n")
    _write(rd / "docker_inspect.json", "{}\n")
    _write(rd / "logs" / "serve.log", "ok\n")
    for gi in gpus:
        _gpu_csv(rd, gi)
    _system_csv(rd)
    _client_csv(rd)
    _agg_csv(rd, unexpected=unexpected, with_col=with_col)
    return rd


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_health(campaign_yaml: Path, runs: Path) -> dict:
    """Drive campaign_health.sh; return {finding_key: (level, message)}."""
    env = dict(os.environ, RUNS_ROOT=str(runs))
    env.pop("MITIGATIONS_LOG", None)
    proc = subprocess.run(["bash", str(HEALTH), "--campaign-yaml", str(campaign_yaml)],
                          capture_output=True, text=True, env=env, timeout=120)
    out = _ANSI.sub("", proc.stdout + proc.stderr)
    findings = {}
    for line in out.splitlines():
        m = _FINDING.match(line)
        if m:
            findings[m.group(2)] = (m.group(1), m.group(3))
    return findings


@unittest.skipUnless(_have_bash(), "bash required")
class SectionBLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cid = "exthc"
        yaml_path, runs = build_campaign(tmp, cid)
        add_single_container(runs, cid, "scvllm")
        add_dynamo(runs, cid, "dynok", shape="launch", unexpected=0)
        add_dynamo(runs, cid, "dyncontam", shape="launch", unexpected=1)
        add_dynamo(runs, cid, "dynnocol", shape="launch", with_col=False)
        add_dynamo(runs, cid, "dynattach", shape="attach", gpus=(2, 3))
        cls.f = run_health(yaml_path, runs)
        cls.cid = cid

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def key(self, run, leaf):
        return f"B.{self.cid}_{run}_r01.{leaf}"

    def no_fail_for(self, run):
        bad = {k: v for k, v in self.f.items()
               if k.startswith(f"B.{self.cid}_{run}_r01.") and v[0] == "FAIL"}
        self.assertEqual(bad, {}, f"unexpected FAILs for {run}: {bad}")

    # ---- single_container path still works end-to-end ----
    def test_single_container_healthy(self):
        self.assertEqual(self.f[self.key("scvllm", "files")][0], "PASS")
        self.assertEqual(self.f[self.key("scvllm", "proc.alive")][0], "PASS")
        self.assertEqual(self.f[self.key("scvllm", "gpu.vram")][0], "PASS")
        # Single-container context carries the single gpu + strategy, no lifecycle.
        self.assertIn("strategy=container_pid1", self.f[self.key("scvllm", "context")][1])
        self.no_fail_for("scvllm")

    # ---- dynamo healthy (launch_cell shape) ----
    def test_dynamo_healthy_all_pass(self):
        ctx = self.f[self.key("dynok", "context")][1]
        self.assertIn("lifecycle=dynamo_disagg", ctx)
        self.assertIn("gpus=0,1", ctx)
        self.assertEqual(self.f[self.key("dynok", "files")][0], "PASS")
        self.assertEqual(self.f[self.key("dynok", "proc.alive")][0], "PASS")
        self.assertEqual(self.f[self.key("dynok", "proc.membership")][0], "PASS")
        # BOTH topology GPUs checked.
        self.assertEqual(self.f[self.key("dynok", "gpu0.vram")][0], "PASS")
        self.assertEqual(self.f[self.key("dynok", "gpu1.vram")][0], "PASS")
        self.no_fail_for("dynok")

    # ---- P1: stray out-of-scope PID (n_pids_unexpected>0) is a hard FAIL ----
    def test_dynamo_contamination_fails_membership_not_alive(self):
        self.assertEqual(self.f[self.key("dyncontam", "proc.membership")][0], "FAIL")
        self.assertIn("n_pids_unexpected", self.f[self.key("dyncontam", "proc.membership")][1])
        # alive stays PASS -- "component present" and "scope clean" are distinct.
        self.assertEqual(self.f[self.key("dyncontam", "proc.alive")][0], "PASS")

    # ---- P1: missing n_pids_unexpected column -> WARN (unverifiable), not silent pass ----
    def test_dynamo_missing_unexpected_column_warns(self):
        self.assertEqual(self.f[self.key("dynnocol", "proc.membership")][0], "WARN")
        self.assertIn("absent", self.f[self.key("dynnocol", "proc.membership")][1])

    # ---- P2 + P3: attach_run shape (identity under monitors.components, ----
    #      engine.lifecycle/topology in JSON, NO cell yaml on disk) ----
    def test_attach_run_shape_identity_and_json_topology(self):
        ctx = self.f[self.key("dynattach", "context")][1]
        # P3: dynamo detected from engine.lifecycle JSON (no top-level, no yaml),
        # topology GPUs read from engine.topology JSON (2,3 -- NOT the 0,1 default).
        self.assertIn("lifecycle=dynamo_disagg", ctx)
        self.assertIn("gpus=2,3", ctx)
        # P2: component identity found under monitors.components.components.
        self.assertEqual(self.f[self.key("dynattach", "files")][0], "PASS")
        self.assertIn("PGID identity", self.f[self.key("dynattach", "files")][1])
        self.assertEqual(self.f[self.key("dynattach", "gpu2.vram")][0], "PASS")
        self.assertEqual(self.f[self.key("dynattach", "gpu3.vram")][0], "PASS")
        self.no_fail_for("dynattach")


@unittest.skipUnless(_have_bash(), "bash required")
class StateCampaignMismatch(unittest.TestCase):
    def test_state_file_campaign_id_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yaml_path, runs = build_campaign(tmp, "exthc")
            # A state file belonging to a DIFFERENT campaign.
            _write(tmp / "camp" / "state" / "campaign_state.json",
                   json.dumps({"campaign_id": "OTHER", "runs": {}}))
            f = run_health(yaml_path, runs)
            self.assertEqual(f["A.state.campaign_id"][0], "FAIL")


@unittest.skipUnless(_have_bash(), "bash required")
class MitigationsWriterReaderContract(unittest.TestCase):
    """log_mitigation.sh must write where campaign_health.sh reads: both derive
    the path from the SAME campaign yaml (the Medium finding)."""

    def test_logged_mitigation_lands_in_campaign_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yaml_path, _ = build_campaign(tmp, "exthc")
            env = dict(os.environ)
            env.pop("MITIGATIONS_LOG", None)
            r = subprocess.run(
                ["bash", str(LOG_MIT), "--campaign-yaml", str(yaml_path),
                 "disk_prune", "reclaimed 22GB"],
                capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            log = tmp / "camp" / "state" / "mitigations.log"
            self.assertTrue(log.exists(), "mitigation not written to the campaign state dir")
            self.assertIn("disk_prune", log.read_text())

    def test_unknown_category_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yaml_path, _ = build_campaign(tmp, "exthc")
            r = subprocess.run(
                ["bash", str(LOG_MIT), "--campaign-yaml", str(yaml_path),
                 "not_a_category", "x"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 1)

    def test_env_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yaml_path, _ = build_campaign(tmp, "exthc")
            override = tmp / "custom.log"
            env = dict(os.environ, MITIGATIONS_LOG=str(override))
            r = subprocess.run(
                ["bash", str(LOG_MIT), "--campaign-yaml", str(yaml_path),
                 "gpu_intervention", "reset gpu0"],
                capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(override.exists())
            self.assertFalse((tmp / "camp" / "state" / "mitigations.log").exists())


if __name__ == "__main__":
    unittest.main()
