"""Process monitor: psutil-based, samples one process by PID at 5 s.

PID resolution. Two modes:

  --pid <int>          monitor exactly this PID
  --pidfile <path>     read the PID from a text file (single int on first line),
                       re-read every period to handle restarts

If the PID disappears, the monitor records a sentinel sample with
process_alive=False and continues. If --pidfile is set and the file
content changes, the monitor switches to the new PID.

Sampled fields per sample:

  ts_unix
  pid
  process_alive
  rss_bytes
  vms_bytes
  uss_bytes              unique set size, often the truest "leak" indicator
  pss_bytes              proportional set size (Linux only)
  num_threads
  num_fds                open file descriptors
  cpu_percent            since last sample
  voluntary_ctx_switches
  voluntary_ctx_switches_rate  switches/sec derived from delta of the counter
  involuntary_ctx_switches
  involuntary_ctx_switches_rate  switches/sec derived from delta of the counter
  io_read_bytes
  io_read_bytes_rate          bytes/sec derived from delta of the counter
  io_write_bytes
  io_write_bytes_rate         bytes/sec derived from delta of the counter
  io_read_count
  io_read_count_rate          ops/sec derived from delta of the counter
  io_write_count
  io_write_count_rate         ops/sec derived from delta of the counter
  num_children           direct child processes
  _sample_duration_s
  _overrun
  _wall_clock_unix
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

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
    "pid",
    "process_alive",
    "rss_bytes",
    "vms_bytes",
    "uss_bytes",
    "pss_bytes",
    "num_threads",
    "num_fds",
    "cpu_percent",
    "voluntary_ctx_switches",
    "voluntary_ctx_switches_rate",
    "involuntary_ctx_switches",
    "involuntary_ctx_switches_rate",
    "io_read_bytes",
    "io_read_bytes_rate",
    "io_write_bytes",
    "io_write_bytes_rate",
    "io_read_count",
    "io_read_count_rate",
    "io_write_count",
    "io_write_count_rate",
    "num_children",
    "_sample_duration_s",
    "_overrun",
    "_wall_clock_unix",
]


def _read_pidfile(path: Path) -> Optional[int]:
    try:
        text = path.read_text().strip().split()[0]
        return int(text)
    except (OSError, ValueError, IndexError):
        return None


class PidResolver:
    def __init__(self, pid: Optional[int], pidfile: Optional[Path]) -> None:
        self.fixed_pid = pid
        self.pidfile = pidfile
        self.last_pid: Optional[int] = pid
        self._proc: Optional[psutil.Process] = None
        if pid is not None:
            self._attach(pid)

    def _attach(self, pid: int) -> None:
        try:
            self._proc = psutil.Process(pid)
            # First call returns 0.0; prime the cpu_percent counter.
            self._proc.cpu_percent(interval=None)
            self.last_pid = pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._proc = None

    def get(self) -> tuple[Optional[psutil.Process], Optional[int]]:
        if self.pidfile is not None:
            current = _read_pidfile(self.pidfile)
            if current is not None and current != self.last_pid:
                self._attach(current)
            elif current is None:
                self._proc = None
                self.last_pid = None
        # Verify the process is still alive.
        if self._proc is not None:
            try:
                if not self._proc.is_running():
                    self._proc = None
            except psutil.Error:
                self._proc = None
        return self._proc, self.last_pid


def make_sampler(resolver: PidResolver):
    previous_sample: Optional[dict[str, Any]] = None
    previous_pid: Optional[int] = None

    def _delta_rate(current: Optional[int], previous: Optional[int], delta_seconds: float) -> Optional[float]:
        if current is None or previous is None or delta_seconds <= 0:
            return float("nan")
        return (current - previous) / delta_seconds

    def sample() -> dict[str, Any]:
        import time

        nonlocal previous_sample, previous_pid

        ts = time.time()
        proc, pid = resolver.get()
        if proc is None:
            return {
                "ts_unix": ts,
                "pid": pid,
                "process_alive": False,
            }

        if pid != previous_pid:
            previous_sample = None

        try:
            with proc.oneshot():
                mem = proc.memory_full_info()
                ctx = proc.num_ctx_switches()
                io = proc.io_counters() if hasattr(proc, "io_counters") else None
                cpu = proc.cpu_percent(interval=None)
                threads = proc.num_threads()
                try:
                    fds = proc.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    fds = None
                try:
                    children = len(proc.children())
                except psutil.Error:
                    children = None

            sample_data = {
                "ts_unix": ts,
                "pid": pid,
                "process_alive": True,
                "rss_bytes": mem.rss,
                "vms_bytes": mem.vms,
                "uss_bytes": getattr(mem, "uss", None),
                "pss_bytes": getattr(mem, "pss", None),
                "num_threads": threads,
                "num_fds": fds,
                "cpu_percent": cpu,
                "voluntary_ctx_switches": ctx.voluntary,
                "involuntary_ctx_switches": ctx.involuntary,
                "io_read_bytes": io.read_bytes if io else None,
                "io_write_bytes": io.write_bytes if io else None,
                "io_read_count": io.read_count if io else None,
                "io_write_count": io.write_count if io else None,
                "num_children": children,
            }

            if previous_sample is None:
                rate_fields = {
                    "voluntary_ctx_switches_rate": float("nan"),
                    "involuntary_ctx_switches_rate": float("nan"),
                    "io_read_bytes_rate": float("nan"),
                    "io_write_bytes_rate": float("nan"),
                    "io_read_count_rate": float("nan"),
                    "io_write_count_rate": float("nan"),
                }
            else:
                delta_seconds = ts - previous_sample["ts_unix"]
                rate_fields = {
                    "voluntary_ctx_switches_rate": _delta_rate(
                        sample_data["voluntary_ctx_switches"],
                        previous_sample["voluntary_ctx_switches"],
                        delta_seconds,
                    ),
                    "involuntary_ctx_switches_rate": _delta_rate(
                        sample_data["involuntary_ctx_switches"],
                        previous_sample["involuntary_ctx_switches"],
                        delta_seconds,
                    ),
                    "io_read_bytes_rate": _delta_rate(
                        sample_data["io_read_bytes"],
                        previous_sample["io_read_bytes"],
                        delta_seconds,
                    ),
                    "io_write_bytes_rate": _delta_rate(
                        sample_data["io_write_bytes"],
                        previous_sample["io_write_bytes"],
                        delta_seconds,
                    ),
                    "io_read_count_rate": _delta_rate(
                        sample_data["io_read_count"],
                        previous_sample["io_read_count"],
                        delta_seconds,
                    ),
                    "io_write_count_rate": _delta_rate(
                        sample_data["io_write_count"],
                        previous_sample["io_write_count"],
                        delta_seconds,
                    ),
                }

            sample_data.update(rate_fields)
            previous_sample = sample_data
            previous_pid = pid
            return sample_data
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "ts_unix": ts,
                "pid": pid,
                "process_alive": False,
            }

    return sample


def main() -> None:
    p = argparse.ArgumentParser(description="Process psutil monitor.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pid", type=int, help="Fixed PID to monitor.")
    g.add_argument("--pidfile", type=Path, help="File containing the PID (re-read on each sample).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="proc", help="Used as base name in the output CSV.")
    p.add_argument("--period-seconds", type=float, default=5.0)
    p.add_argument("--rotation-seconds", type=int, default=60)
    p.add_argument("--watchdog-timeout-s", type=float, default=30.0)
    p.add_argument(
        "--duration-seconds",
        type=int,
        default=0,
        help="If >0, auto-exit after this many seconds. Lets bash drivers run "
             "a bounded smoke without sending SIGTERM/SIGINT to a root process.",
    )
    args = p.parse_args()

    resolver = PidResolver(pid=args.pid, pidfile=args.pidfile)
    sampler = make_sampler(resolver)
    watchdog = Watchdog(timeout_seconds=args.watchdog_timeout_s, sentinel={"process_alive": False})
    writer = CsvRotatingWriter(
        WriterConfig(
            output_dir=args.output_dir,
            base_name=args.label,
            rotation_seconds=args.rotation_seconds,
            fieldnames=FIELDNAMES,
        )
    )
    shutdown = ShutdownEvent()

    # Auto-shutdown timer (background thread). Lets a non-root caller run a
    # bounded proc_monitor under sudo without needing kill privileges.
    if args.duration_seconds > 0:
        import threading
        import time as _time

        def _stopper():
            _time.sleep(args.duration_seconds)
            shutdown._event.set()

        threading.Thread(target=_stopper, daemon=True).start()

    def guarded_sample():
        return watchdog.call(sampler)

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
