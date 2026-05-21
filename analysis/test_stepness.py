import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stepness  # noqa: E402


ONE_MIB = 1024 ** 2


class StepnessEdgeCaseTest(unittest.TestCase):
    def analyze_synthetic(self, frame):
        patches = [
            patch.object(stepness, "load_manifest", return_value={}),
            patch.object(stepness, "discover_proc_prefix", return_value="proc"),
            patch.object(stepness, "load_proc", return_value=frame.copy()),
            patch.object(stepness, "resolve_warmup", return_value=0),
            patch.object(stepness, "infer_cell_id", return_value="synthetic"),
        ]
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            return stepness.analyze_run(
                Path("/tmp/synthetic_stepness"),
                warmup_s=0,
                n_bootstrap=1,
                seed=1,
            )

    def test_top1_positive_jumps_ignores_zero_mass(self):
        arr = np.array([0.0] * 1000 + [ONE_MIB, 2 * ONE_MIB, 3 * ONE_MIB])
        metrics = stepness._compute_kurtosis_metrics(
            arr,
            n_bootstrap=1,
            rng=np.random.default_rng(1),
            basename="synthetic_top1",
            label="RSS",
        )
        self.assertEqual(metrics["mean_top1_step_mb"], 3.0)

    def test_pid_transition_is_not_counted_as_step(self):
        frame = pd.DataFrame(
            {
                "ts_unix": [0.0, 10.0, 20.0, 30.0],
                "pid": [111, 111, 222, 222],
                "rss_bytes": [
                    100 * ONE_MIB,
                    100.5 * ONE_MIB,
                    1000 * ONE_MIB,
                    1000.5 * ONE_MIB,
                ],
                "vms_bytes": [
                    200 * ONE_MIB,
                    200.5 * ONE_MIB,
                    2000 * ONE_MIB,
                    2000.5 * ONE_MIB,
                ],
            }
        )
        result = self.analyze_synthetic(frame)
        self.assertEqual(result["steps_per_h_1mb"], 0.0)
        self.assertEqual([round(x / ONE_MIB, 3) for x in result["_diff_rss"]], [0.5, 0.5])

    def test_top_k_timestamps_follow_masked_diff_indices(self):
        frame = pd.DataFrame(
            {
                "ts_unix": [100.0, 110.0, 120.0, 130.0],
                "pid": [1, 1, 2, 2],
                "rss_bytes": [100 * ONE_MIB, 101 * ONE_MIB, 1000 * ONE_MIB, 1005 * ONE_MIB],
                "vms_bytes": [200 * ONE_MIB, 201 * ONE_MIB, 2000 * ONE_MIB, 2005 * ONE_MIB],
            }
        )
        result = self.analyze_synthetic(frame)
        self.assertEqual([float(x) for x in result["_diff_ts"]], [110.0, 130.0])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stepness.print_top_k([result], 1)
        self.assertIn("1970-01-01 00:02:10 UTC", out.getvalue())

    def test_all_nan_vms_is_unusable_not_drift(self):
        frame = pd.DataFrame(
            {
                "ts_unix": [0.0, 10.0, 20.0, 30.0, 40.0],
                "pid": [1, 1, 1, 1, 1],
                "rss_bytes": [100 * ONE_MIB] * 5,
                "vms_bytes": [np.nan] * 5,
            }
        )
        result = self.analyze_synthetic(frame)
        self.assertEqual(result["class"], "border")
        self.assertIn("VMS_unusable", result["notes"])
        self.assertTrue(np.isnan(result["steps_per_h_1mb_dVMS"]))

    def test_sparse_bootstrap_filters_undefined_resamples(self):
        arr = np.zeros(100)
        arr[-1] = 5 * ONE_MIB
        lo, hi = stepness.bootstrap_ci(arr, 200, np.random.default_rng(123))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))

    def test_bootstrap_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            stepness.bootstrap_ci(np.arange(10.0), 0, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
