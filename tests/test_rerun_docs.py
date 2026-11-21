"""R9: docs/rerun_windows.md stays consistent with the code and the descriptors.

Guards against drift: the CP-correction assumption is stated, every operator
step has acceptance criteria, every referenced script exists, the close reason
is the agreed one, and the projection table matches the generated descriptors.

Run: python3 -m unittest tests.test_rerun_docs
"""
import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "rerun_windows.md"
EXT = REPO / "campaigns" / "extension"


class RerunDoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOC.read_text()

    def test_assumption_is_stated_explicitly(self):
        self.assertIn("**Assumption W.**", self.text)
        flat = re.sub(r"\s*\n>?\s*", " ", self.text)   # unwrap lines and blockquotes
        for phrase in ("additive", "no window x factor interaction", "same for Triton and vLLM",
                       "cannot detect or remove a window x factor interaction"):
            self.assertIn(phrase, flat)

    def test_blocking_guarantee_is_scoped_to_additive_offsets(self):
        flat = re.sub(r"\s*\n>?\s*", " ", self.text)
        self.assertNotIn("whatever its form", flat)
        for phrase in ("additive on the analysis scale",
                       "a window x A interaction produces an apparent A main effect",
                       "cannot detect or remove a window x factor interaction",
                       "for the main effects as much as for AC, AD and CD",
                       "needs all six center points", "INCOMPLETE, never CONSISTENT"):
            self.assertIn(phrase, flat)

    def test_operator_sequence_with_acceptance_per_step(self):
        steps = ["**1. Reboot.**", "**2. Environment snapshot.**", "**3. Soak, W1 only",
                 "**4. Calibration of window k.**", "**5. Dry-run of the window.**",
                 "**6. Start.**", "**7. Status", "**8. Close-out of the window.**"]
        positions = [self.text.index(s) for s in steps]
        self.assertEqual(positions, sorted(positions))
        bounds = positions + [self.text.index("### After W3")]
        for step, (a, b) in zip(steps, zip(bounds, bounds[1:])):
            self.assertIn("Acceptance", self.text[a:b], step)

    def test_close_reason_is_the_agreed_one(self):
        self.assertIn('--reason "superseded: incomplete Triton/vLLM process scope, '
                      'rerun under extension_dow_rescoped"', self.text)

    def test_referenced_scripts_and_descriptors_exist(self):
        for rel in set(re.findall(r"((?:scripts|analysis)/[\w/]+\.(?:py|sh))", self.text)):
            self.assertTrue((REPO / rel).exists(), rel)
        for name in set(re.findall(r"`(dow_rerun_w\d(?:_soak)?\.yaml)`", self.text)):
            self.assertTrue((EXT / name).exists(), name)

    def test_projection_table_matches_descriptors(self):
        for k in (1, 2, 3):
            meta = yaml.safe_load((EXT / f"dow_rerun_w{k}.yaml").read_text())["rerun_window"]
            row = next(line for line in self.text.splitlines() if line.startswith(f"| W{k} |"))
            self.assertIn(f"{meta['projected_total_days']:.1f} d", row)
            self.assertIn(f"{meta['projected_retry_margin_days']:.1f} d", row)
            self.assertIn(f"{meta['projected_calibration_h']:.1f} h", row)
            self.assertIn(f"~{meta['projected_calibration_age_at_last_dispatch_days']:.1f} d", row)


if __name__ == "__main__":
    unittest.main()
