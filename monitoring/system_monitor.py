"""System-wide monitor: global resources at 5 s.

Sampled fields:

  ts_unix
  mem_total_bytes
  mem_available_bytes
  mem_used_bytes
  mem_free_bytes
  mem_cached_bytes
  mem_buffers_bytes
  swap_total_bytes
  swap_used_bytes
  swap_free_bytes
  swap_in_bytes_cumulative      cumulative since boot
  swap_out_bytes_cumulative
  load_avg_1m
  load_avg_5m
  load_avg_15m
  cpu_percent_total
  cpu_percent_iowait
  fd_allocated                  /proc/sys/fs/file-nr first column
  fd_max                        /proc/sys/fs/file-nr third column
  disk_read_bytes_cumulative    sum across all physical disks
  disk_write_bytes_cumulative
  net_rx_bytes_cumulative       sum across all interfaces
  net_tx_bytes_cumulative
  _sample_duration_s
  _overrun
  _wall_clock_unix

Disk and network counters are reported as cumulative; rates are derived
in post-processing. This avoids state in the monitor and keeps the
sampling stateless.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil not installed. Run: pip install psutil") from e

from _common import (
    CsvRotatingWriter,
    ShutdownEvent,
    Watchdog,
    WriterConfig,
    steady_sampler,
)


FIELDNAMES = [
    "ts_unix",
    "mem_total_bytes",
    "mem_available_bytes",
    "mem_used_bytes",
    "mem_free_bytes",
    "mem_cached_bytes",
    "mem_buffers_bytes",
    "swap_total_bytes",
    "swap_used_bytes",
    "swap_free_bytes",
    "swap_in_bytes_cumulative",
    "swap_out_bytes_cumulative",
    "load_avg_1m",
    "load_avg_5m",
    "load_avg_15m",
    "cpu_percent_total",
    "cpu_percent_iowait",
    "fd_allocated",
    "fd_max",
    "disk_read_bytes_cumulative",
    "disk_write_bytes_cumulative",
    "net_rx_bytes_cumulative",
    "net_tx_bytes_cumulative",
    "_sample_duration_s",
    "_overrun",
    "_wall_clock_unix",
]


def _read_file_nr() -> tuple[int | None, int | None]:
    try:
        parts = Path("/proc/sys/fs/file-nr").read_text().split()
        return int(parts[0]), int(parts[2])
    except (OSError, ValueError, IndexError):
        return None, None


def sample() -> dict[str, Any]:
    import time

    ts = time.time()
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    la = psutil.getloadavg()
    cpu_times = psutil.cpu_times_percent(interval=None)
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    fd_alloc, fd_max = _read_file_nr()

    return {
        "ts_unix": ts,
        "mem_total_bytes": vm.total,
        "mem_available_bytes": vm.available,
        "mem_used_bytes": vm.used,
        "mem_free_bytes": vm.free,
        "mem_cached_bytes": getattr(vm, "cached", None),
        "mem_buffers_bytes": getattr(vm, "buffers", None),
        "swap_total_bytes": sw.total,
        "swap_used_bytes": sw.used,
        "swap_free_bytes": sw.free,
        "swap_in_bytes_cumulative": getattr(sw, "sin", None),
        "swap_out_bytes_cumulative": getattr(sw, "sout", None),
        "load_avg_1m": la[0],
        "load_avg_5m": la[1],
        "load_avg_15m": la[2],
        "cpu_percent_total": 100.0 - cpu_times.idle,
        "cpu_percent_iowait": getattr(cpu_times, "iowait", None),
        "fd_allocated": fd_alloc,
        "fd_max": fd_max,
        "disk_read_bytes_cumulative": disk.read_bytes if disk else None,
        "disk_write_bytes_cumulative": disk.write_bytes if disk else None,
        "net_rx_bytes_cumulative": net.bytes_recv if net else None,
        "net_tx_bytes_cumulative": net.bytes_sent if net else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="System-wide resource monitor.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--period-seconds", type=float, default=5.0)
    p.add_argument("--rotation-seconds", type=int, default=60)
    p.add_argument("--watchdog-timeout-s", type=float, default=30.0)
    args = p.parse_args()

    # Prime cpu_times_percent so the first real sample has a valid delta.
    psutil.cpu_times_percent(interval=None)

    watchdog = Watchdog(timeout_seconds=args.watchdog_timeout_s, sentinel={})
    writer = CsvRotatingWriter(
        WriterConfig(
            output_dir=args.output_dir,
            base_name="system",
            rotation_seconds=args.rotation_seconds,
            fieldnames=FIELDNAMES,
        )
    )
    shutdown = ShutdownEvent()

    def guarded_sample():
        return watchdog.call(sample)

    try:
        steady_sampler(
            sampler=guarded_sample,
            period_seconds=args.period_seconds,
            shutdown=shutdown,
            on_sample=writer.write,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
