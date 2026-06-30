"""PHASE A point 5 (hardening): free_gb never fabricates inf, and an unknown /
unstattable filesystem is a HARD FAIL in require_free_space.
Run: python3 -m unittest tests.test_phase_a_diskgate
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import launch_cell as lc  # noqa: E402


class _Died(Exception):
    pass


def _die(msg, rc=1):
    raise _Died(f"{rc}:{msg}")


class FreeGb(unittest.TestCase):
    def test_valid_path_returns_positive_float(self):
        g = lc.free_gb(Path("/"))
        self.assertIsInstance(g, float)
        self.assertGreater(g, 0.0)

    def test_bad_path_returns_none_not_inf(self):
        g = lc.free_gb(Path("/no/such/path/wosar_xyz"))
        self.assertIsNone(g)  # crucially NOT float('inf')


class RequireFreeSpace(unittest.TestCase):
    def test_none_path_is_hard_fail(self):
        with patch.object(lc, "die", _die):
            with self.assertRaises(_Died):
                lc.require_free_space([None], min_gb=1.0)

    def test_unstattable_path_is_hard_fail(self):
        with patch.object(lc, "die", _die):
            with self.assertRaises(_Died):
                lc.require_free_space([Path("/no/such/path/wosar_xyz")], min_gb=1.0)

    def test_below_min_is_hard_fail(self):
        with patch.object(lc, "die", _die):
            with self.assertRaises(_Died):
                lc.require_free_space([Path("/")], min_gb=1.0e12)  # 1 EB, never satisfied

    def test_enough_space_passes(self):
        with patch.object(lc, "die", _die):
            lc.require_free_space([Path("/")], min_gb=0.001)  # must not raise


if __name__ == "__main__":
    unittest.main()
