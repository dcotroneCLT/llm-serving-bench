import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "psutil" not in sys.modules:
    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=Exception,
        AccessDenied=Exception,
        Error=Exception,
    )
    sys.modules["psutil"] = fake_psutil

import proc_monitor  # noqa: E402


class DummyProcess:
    def __init__(self, samples):
        self._samples = iter(samples)
        self.current = None

    def oneshot(self):
        return self

    def __enter__(self):
        self.current = next(self._samples)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def memory_full_info(self):
        return SimpleNamespace(
            rss=self.current["rss_bytes"],
            vms=self.current["vms_bytes"],
            uss=self.current["uss_bytes"],
            pss=self.current["pss_bytes"],
        )

    def num_ctx_switches(self):
        return SimpleNamespace(
            voluntary=self.current["voluntary_ctx_switches"],
            involuntary=self.current["involuntary_ctx_switches"],
        )

    def io_counters(self):
        return SimpleNamespace(
            read_bytes=self.current["io_read_bytes"],
            write_bytes=self.current["io_write_bytes"],
            read_count=self.current["io_read_count"],
            write_count=self.current["io_write_count"],
        )

    def cpu_percent(self, interval=None):
        return self.current["cpu_percent"]

    def num_threads(self):
        return self.current["num_threads"]

    def num_fds(self):
        return self.current["num_fds"]

    def children(self):
        return [None] * self.current["num_children"]

    def is_running(self):
        return True


class DummyResolver:
    def __init__(self, process, pid):
        self.process = process
        self.pid = pid

    def get(self):
        return self.process, self.pid


class ProcMonitorRateTest(unittest.TestCase):
    def test_stable_process_rates_are_positive(self):
        process_samples = [
            {
                "rss_bytes": 1000,
                "vms_bytes": 2000,
                "uss_bytes": 500,
                "pss_bytes": 600,
                "num_threads": 1,
                "num_fds": 3,
                "cpu_percent": 0.0,
                "voluntary_ctx_switches": 10,
                "involuntary_ctx_switches": 5,
                "io_read_bytes": 1000,
                "io_write_bytes": 2000,
                "io_read_count": 10,
                "io_write_count": 20,
                "num_children": 0,
            },
            {
                "rss_bytes": 1100,
                "vms_bytes": 2100,
                "uss_bytes": 510,
                "pss_bytes": 610,
                "num_threads": 1,
                "num_fds": 3,
                "cpu_percent": 0.0,
                "voluntary_ctx_switches": 20,
                "involuntary_ctx_switches": 10,
                "io_read_bytes": 2000,
                "io_write_bytes": 3000,
                "io_read_count": 20,
                "io_write_count": 30,
                "num_children": 0,
            },
            {
                "rss_bytes": 1200,
                "vms_bytes": 2200,
                "uss_bytes": 520,
                "pss_bytes": 620,
                "num_threads": 1,
                "num_fds": 3,
                "cpu_percent": 0.0,
                "voluntary_ctx_switches": 30,
                "involuntary_ctx_switches": 15,
                "io_read_bytes": 3000,
                "io_write_bytes": 4000,
                "io_read_count": 30,
                "io_write_count": 40,
                "num_children": 0,
            },
        ]
        dummy_process = DummyProcess(process_samples)
        resolver = DummyResolver(dummy_process, pid=1234)
        sampler = proc_monitor.make_sampler(resolver)

        with patch("time.time", side_effect=[0.0, 5.0, 10.0]):
            sample1 = sampler()
            sample2 = sampler()
            sample3 = sampler()

        self.assertFalse(sample1["voluntary_ctx_switches_rate"] == sample1["voluntary_ctx_switches_rate"])
        self.assertFalse(sample1["io_read_bytes_rate"] == sample1["io_read_bytes_rate"])

        self.assertGreater(sample2["voluntary_ctx_switches_rate"], 0)
        self.assertGreater(sample2["involuntary_ctx_switches_rate"], 0)
        self.assertGreater(sample2["io_read_bytes_rate"], 0)
        self.assertGreater(sample2["io_write_bytes_rate"], 0)
        self.assertGreater(sample2["io_read_count_rate"], 0)
        self.assertGreater(sample2["io_write_count_rate"], 0)

        self.assertGreater(sample3["voluntary_ctx_switches_rate"], 0)
        self.assertGreater(sample3["involuntary_ctx_switches_rate"], 0)
        self.assertGreater(sample3["io_read_bytes_rate"], 0)
        self.assertGreater(sample3["io_write_bytes_rate"], 0)
        self.assertGreater(sample3["io_read_count_rate"], 0)
        self.assertGreater(sample3["io_write_count_rate"], 0)


if __name__ == "__main__":
    unittest.main()
