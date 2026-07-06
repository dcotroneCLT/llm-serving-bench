"""validation_check RSS slope input (DEFECT 1).

Verified facts this pins down:
  * The full rotated proc series is assembled from ALL segments before the
    warmup/windowing runs (already true via load_proc_concat; a regression to
    single-segment reading would fail test_full_series_used_across_segments).
  * The warmup discard honors the run's declared warmup_discard_s (what the
    aging_io/aging_trends pipeline uses), not a fixed 1800/3600 heuristic that
    exceeded a short validation run's span and discarded every sample.
  * For runs whose manifest lacks warmup_discard_s (older layout) OR whose
    declared warmup equals the legacy heuristic, output is unchanged.

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_validation_check
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))

import validation_check as vc  # noqa: E402


LABEL = "vllm_standalone_020"


def _rss(i: int) -> int:
    # Monotone climb with a small deterministic wiggle so Mann-Kendall has
    # variance to work with (a perfectly linear series yields nan z/p).
    return 1_000_000_000 + 200_000 * i + 30_000 * ((i % 5) - 2)


def build_run(tmp: Path, n_segments: int, seg_samples: int,
              warmup_discard_s=None, start=1_700_000_000.0) -> Path:
    run = tmp / "extension_dow_val_vllm_r01"
    (run / "client").mkdir(parents=True)
    i = 0
    t = start + 0.5
    for seg in range(n_segments):
        lines = ["ts_unix,rss_bytes,uss_bytes,process_alive"]
        for _ in range(seg_samples):
            lines.append(f"{t:.3f},{_rss(i)},{_rss(i) - 50_000_000},True")
            t += 1.0
            i += 1
        (run / f"{LABEL}_{seg:06d}.csv").write_text("\n".join(lines) + "\n")
    span = t - (start + 0.5)
    (run / "gpu0_000000.csv").write_text(f"ts_unix,gpu_util\n{start},10\n")
    (run / "system_000000.csv").write_text(f"ts_unix,cpu\n{start},1\n")
    (run / "client" / "requests_000000.csv").write_text("req_id,status\n0,ok\n1,ok\n")
    manifest = {
        "run_id": "extension_dow_val_vllm_r01", "cell_id": "val_vllm", "replica": 1,
        "image": {"digest": "sha256:x"}, "started_at": "t0", "ended_at": "t1",
        "duration_seconds_actual": span, "duration_s": int(span),
        "interrupted_early": False, "proc_prefix": LABEL,
        "monitors": {"proc": {"label": LABEL}},
    }
    if warmup_discard_s is not None:
        manifest["warmup_discard_s"] = warmup_discard_s
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def run_check(run: Path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = vc.check(run)
    return rc, buf.getvalue()


def parse_n_used(out: str):
    m = re.search(r"n samples used\s*:\s*(\d+)", out)
    return int(m.group(1)) if m else None


class FullSeriesAcrossSegments(unittest.TestCase):
    def test_full_series_used_across_segments(self):
        # 3 segments of 60s (span 180s); declared warmup 120s. Segment 0 (0-60s)
        # falls ENTIRELY inside the warmup: a single-segment reader would leave
        # zero post-warmup samples and HARD-fail. The full series survives with
        # the samples from segments spanning 120-180s.
        with tempfile.TemporaryDirectory() as tmp:
            run = build_run(Path(tmp), n_segments=3, seg_samples=60,
                            warmup_discard_s=120)
            rc, out = run_check(run)
            self.assertNotIn("insufficient samples after warmup discard", out)
            n_used = parse_n_used(out)
            self.assertIsNotNone(n_used)
            # Post-120s of a 180s run -> ~60 samples, far more than the 0 a
            # single-segment (segment 0 only) read would yield.
            self.assertGreaterEqual(n_used, 30)
            self.assertLess(rc, vc.HARD_FAIL)

    def test_short_validation_run_no_longer_hard_fails(self):
        # The reported box case: ~1200s run, declared warmup 120s. The old fixed
        # 1800s heuristic discarded everything; now it is judged.
        with tempfile.TemporaryDirectory() as tmp:
            run = build_run(Path(tmp), n_segments=19, seg_samples=63,
                            warmup_discard_s=120)
            rc, out = run_check(run)
            self.assertNotIn("RSS slope test failed", out)
            self.assertEqual(parse_n_used(out), 19 * 63 - 120)


class ByteIdenticalLegacy(unittest.TestCase):
    def _expected_slope(self, run: Path, warmup: int):
        import pandas as pd
        df = pd.concat([pd.read_csv(p) for p in vc.find_proc_csvs(run)],
                       ignore_index=True).sort_values("ts_unix")
        return vc.rss_slope_mb_per_h(df["rss_bytes"].values.astype(float),
                                     df["ts_unix"].values.astype(float), warmup)

    def test_manifest_absent_falls_back_to_heuristic(self):
        # No warmup_discard_s in manifest, span 2000s (< 2h) -> legacy 1800s.
        with tempfile.TemporaryDirectory() as tmp:
            run = build_run(Path(tmp), n_segments=1, seg_samples=2000,
                            warmup_discard_s=None)
            rc, out = run_check(run)
            exp = self._expected_slope(run, 1800)  # what the legacy heuristic used
            self.assertEqual(parse_n_used(out), exp["n_used"])

    def test_declared_warmup_equal_to_heuristic_is_unchanged(self):
        # A manifest that records the same value the heuristic would pick yields
        # byte-identical windowing (the 36h baseline: warmup 3600, span > 2h).
        with tempfile.TemporaryDirectory() as tmp:
            run = build_run(Path(tmp), n_segments=1, seg_samples=2000,
                            warmup_discard_s=1800)
            rc, out = run_check(run)
            exp = self._expected_slope(run, 1800)
            self.assertEqual(parse_n_used(out), exp["n_used"])


if __name__ == "__main__":
    unittest.main()
