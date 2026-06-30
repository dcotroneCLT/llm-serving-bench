"""Guards for the consolidated box-validation runbook (repass_gate2.sh) and the
empirical scoping check (verify_scoping.py).
Run: python3 -m unittest tests.test_phase_a_repass
"""
import importlib.util
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DYN = REPO / "deploy" / "dynamo"
REPASS = DYN / "repass_gate2.sh"

SUMMARY_KEYS = ["pip_pin", "bringup", "validator", "no_orphans",
                "n_pids_unexpected_0", "verify_scoping", "fail_loud_negative", "disk_root"]


def _load_verify_scoping():
    spec = importlib.util.spec_from_file_location("verify_scoping", DYN / "verify_scoping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RepassRunbook(unittest.TestCase):
    def test_parses(self):
        r = subprocess.run(["bash", "-n", str(REPASS)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_cleans_up_on_exit(self):
        txt = REPASS.read_text()
        self.assertIn("trap cleanup EXIT", txt)
        for tool in ("serve_down.sh", "infra_down.sh", "reaper"):
            self.assertIn(tool, txt, f"cleanup must invoke {tool}")

    def test_has_all_summary_checks(self):
        txt = REPASS.read_text()
        for k in SUMMARY_KEYS:
            self.assertIn(k, txt, f"summary key {k} missing from the runbook")


class VerifyScoping(unittest.TestCase):
    def setUp(self):
        self.vs = _load_verify_scoping()

    def test_broad_regex_covers_all_components(self):
        for cmd in [
            "python -m dynamo.frontend --http-port 8400",
            "python -m dynamo.vllm --model Q --disaggregation-mode prefill",
            "python -m dynamo.vllm --model Q --disaggregation-mode decode",
            "/usr/local/bin/etcd --listen-client-urls http://0.0.0.0:2379",
            "nats-server -js -p 4222",
        ]:
            self.assertTrue(self.vs.BROAD.search(cmd), f"BROAD must match: {cmd}")

    def test_broad_regex_ignores_unrelated(self):
        self.assertIsNone(self.vs.BROAD.search("python -m http.server 8000"))

    def test_has_tolerance_args(self):
        # the empirical comparison must be tunable, not hard-coded zero
        src = (DYN / "verify_scoping.py").read_text()
        self.assertIn("--tolerance-mb", src)
        self.assertIn("--tolerance-frac", src)
        self.assertIn("all-dynamo-PID", src)  # the B) smaps total vs A) aggregate


if __name__ == "__main__":
    unittest.main()
