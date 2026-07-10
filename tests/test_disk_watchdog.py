"""Mid-run disk watchdog helpers (SC-2, PHASE B item 5).

Pure-function coverage for launch_cell's disk watchdog: floor-breach detection
on either filesystem (fail-closed on an unstat-able path), the trend snapshot,
and the additive disk_usage.csv writer. The end-to-end exit-7 / graceful
teardown / interruption_reason wiring is covered in test_phase_b_reaper.py
(LaunchCellWiring).

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_disk_watchdog
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import launch_cell as lc  # noqa: E402


class DiskWatchdogReason(unittest.TestCase):
    def test_healthy_both_filesystems_returns_none(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            # Real dirs, floor 0 GB -> real free space is above it.
            self.assertIsNone(lc.disk_watchdog_reason(Path(a), Path(b), 0.0))

    def test_runs_root_breach_detected(self):
        free = {"/runs": 1.0, "/docker": 500.0}
        with mock.patch.object(lc, "free_gb", side_effect=lambda p: free[str(p)]):
            reason = lc.disk_watchdog_reason(Path("/runs"), Path("/docker"), 10.0)
        self.assertIsNotNone(reason)
        self.assertIn("runs-root", reason)
        self.assertTrue(reason.startswith("disk_floor"))

    def test_docker_root_breach_detected(self):
        # runs-root healthy, docker data-root below the floor -> EITHER trips it.
        free = {"/runs": 500.0, "/docker": 1.0}
        with mock.patch.object(lc, "free_gb", side_effect=lambda p: free[str(p)]):
            reason = lc.disk_watchdog_reason(Path("/runs"), Path("/docker"), 10.0)
        self.assertIsNotNone(reason)
        self.assertIn("docker-root", reason)

    def test_unstattable_is_fail_closed(self):
        # free_gb returns None on stat failure; an unknown disk must be a breach.
        with mock.patch.object(lc, "free_gb", return_value=None):
            reason = lc.disk_watchdog_reason(Path("/runs"), Path("/docker"), 10.0)
        self.assertIsNotNone(reason)
        self.assertIn("cannot stat", reason)

    def test_docker_root_none_only_checks_runs_root(self):
        with mock.patch.object(lc, "free_gb", return_value=500.0) as fg:
            self.assertIsNone(lc.disk_watchdog_reason(Path("/runs"), None, 10.0))
        # Only the runs-root was probed (docker root unknown -> not fabricated).
        self.assertEqual(fg.call_count, 1)


class InodeFloor(unittest.TestCase):
    # Item 4c: the mid-run watchdog also gates on free inodes (statvfs f_favail).
    def test_no_inode_floor_when_arg_omitted(self):
        # Space is fine; inodes are never probed unless a floor is passed.
        with mock.patch.object(lc, "free_gb", return_value=500.0), \
             mock.patch.object(lc, "free_inodes") as fi:
            self.assertIsNone(lc.disk_watchdog_reason(Path("/runs"), None, 10.0))
        fi.assert_not_called()

    def test_inode_breach_detected(self):
        with mock.patch.object(lc, "free_gb", return_value=500.0), \
             mock.patch.object(lc, "free_inodes", return_value=1000):
            reason = lc.disk_watchdog_reason(Path("/runs"), None, 10.0,
                                             min_free_inodes=100000)
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("inode_floor"))
        self.assertIn("runs-root", reason)

    def test_inode_ok_passes(self):
        with mock.patch.object(lc, "free_gb", return_value=500.0), \
             mock.patch.object(lc, "free_inodes", return_value=5_000_000):
            self.assertIsNone(lc.disk_watchdog_reason(Path("/runs"), None, 10.0,
                                                      min_free_inodes=100000))

    def test_inode_none_skips_that_fs(self):
        # An inode-less filesystem (free_inodes -> None) is skipped, not failed:
        # the byte-space side already fails-closed on a truly unstat-able path.
        with mock.patch.object(lc, "free_gb", return_value=500.0), \
             mock.patch.object(lc, "free_inodes", return_value=None):
            self.assertIsNone(lc.disk_watchdog_reason(Path("/runs"), None, 10.0,
                                                      min_free_inodes=100000))

    def test_space_breach_takes_precedence_over_inode_check(self):
        # A byte-space breach returns before inodes are ever probed.
        with mock.patch.object(lc, "free_gb", return_value=1.0), \
             mock.patch.object(lc, "free_inodes") as fi:
            reason = lc.disk_watchdog_reason(Path("/runs"), None, 10.0,
                                             min_free_inodes=100000)
        self.assertTrue(reason.startswith("disk_floor"))
        fi.assert_not_called()

    def test_free_inodes_real_dir_is_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = lc.free_inodes(Path(tmp))
            # Either a real positive count, or None on an inode-less backing fs.
            self.assertTrue(n is None or n > 0)

    def test_free_inodes_missing_path_is_none(self):
        self.assertIsNone(lc.free_inodes(Path("/no/such/path/xyz")))


class DirSizeMb(unittest.TestCase):
    def test_sums_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.bin").write_bytes(b"x" * (1024 * 1024))       # 1 MB
            (root / "sub").mkdir()
            (root / "sub" / "b.bin").write_bytes(b"y" * (512 * 1024))  # 0.5 MB
            self.assertAlmostEqual(lc.dir_size_mb(root), 1.5, places=3)

    def test_empty_dir_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(lc.dir_size_mb(Path(tmp)), 0.0)


class DiskUsageSnapshotAndCsv(unittest.TestCase):
    def test_snapshot_fields(self):
        free = {"/runs": 123.456, "/docker": 78.9}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "f.bin").write_bytes(b"z" * (1024 * 1024))
            with mock.patch.object(lc, "free_gb", side_effect=lambda p: free.get(str(p))):
                snap = lc.disk_usage_snapshot(Path("/runs"), Path("/docker"), run_dir, 1700.5)
        self.assertEqual(set(snap), set(lc.DISK_USAGE_FIELDS))
        self.assertEqual(snap["ts_unix"], 1700.5)
        self.assertEqual(snap["runs_root_free_gb"], 123.46)
        self.assertEqual(snap["docker_root_free_gb"], 78.9)
        self.assertAlmostEqual(snap["run_dir_size_mb"], 1.0, places=1)

    def test_snapshot_none_docker_root_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(lc, "free_gb", return_value=None):
                snap = lc.disk_usage_snapshot(Path("/runs"), None, Path(tmp), 1.0)
        self.assertEqual(snap["docker_root_free_gb"], "")
        self.assertEqual(snap["runs_root_free_gb"], "")  # None -> blank, never fabricated

    def test_append_writes_header_once_then_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "disk_usage.csv"
            row = {"ts_unix": 1.0, "runs_root_free_gb": 50.0,
                   "docker_root_free_gb": 60.0, "run_dir_size_mb": 12.3}
            lc.append_disk_usage_row(csv_path, row)
            lc.append_disk_usage_row(csv_path, {**row, "ts_unix": 2.0})
            lines = csv_path.read_text().splitlines()
        self.assertEqual(lines[0], ",".join(lc.DISK_USAGE_FIELDS))  # header once
        self.assertEqual(len(lines), 3)                             # header + 2 rows
        self.assertEqual(lines[1], "1.0,50.0,60.0,12.3")
        self.assertEqual(lines[2].split(",")[0], "2.0")


if __name__ == "__main__":
    unittest.main()
