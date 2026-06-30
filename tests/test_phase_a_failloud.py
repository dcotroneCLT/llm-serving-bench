"""PHASE A point 1: fail-loud bring-up. Static + syntax guards that actually run
without docker/GPU (off-box). Run: python3 -m unittest tests.test_phase_a_failloud
"""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DYN = REPO / "deploy" / "dynamo"
SCRIPTS = ["env.sh", "infra_up.sh", "infra_down.sh", "serve_down.sh",
           "serve_aggregated.sh", "serve_disaggregated.sh"]


class FailLoudBringUp(unittest.TestCase):
    def test_scripts_parse(self):
        for s in SCRIPTS:
            r = subprocess.run(["bash", "-n", str(DYN / s)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s} failed bash -n: {r.stderr}")

    def test_errexit_enabled(self):
        for s in SCRIPTS:
            txt = (DYN / s).read_text()
            self.assertIn("set -euo pipefail", txt, f"{s} must enable errexit")
            self.assertNotIn("set -uo pipefail", txt, f"{s} still has the non-errexit set line")

    def test_serve_scripts_exit_on_unserved(self):
        for s in ["serve_aggregated.sh", "serve_disaggregated.sh"]:
            txt = (DYN / s).read_text()
            self.assertNotIn("WARNING: model not served", txt,
                             f"{s} still only WARNS on an unserved model")
            self.assertIn("exit 1", txt, f"{s} must exit non-zero on an unserved model")
            self.assertIn('if [ "$served" != 1 ]', txt, f"{s} missing the fail-loud served guard")

    def test_verify_scoping_exits_on_incomplete(self):
        txt = (DYN / "verify_scoping.py").read_text()
        self.assertIn("sys.exit(2)", txt,
                      "verify_scoping.py must exit non-zero on INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
