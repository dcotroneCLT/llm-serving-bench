"""Common utilities for monitoring agents.

CsvRotatingWriter buffers samples and flushes them to a fresh CSV file
every `rotation_seconds`. This bounds data loss to the rotation interval
in case of crash. Files are named with the run_id and a monotonic
sequence number; a manifest file lists all files produced by the writer.

Watchdog runs a callable on a separate thread; if the call exceeds
`timeout_seconds`, a sentinel value is returned and the original call is
abandoned. Used to defend against pynvml or psutil deadlocks under
heavy contention.

ShutdownEvent provides a unified, signal-aware way to request graceful
shutdown. Each monitor wires it up to SIGTERM and SIGINT.
"""

from __future__ import annotations

import csv
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


@dataclass
class WriterConfig:
    output_dir: Path
    base_name: str
    rotation_seconds: int = 60
    fieldnames: Optional[list[str]] = None


class CsvRotatingWriter:
    """Append-only CSV writer with time-based rotation.

    Samples are written to a current file; every rotation_seconds a new
    file is opened (sequence number incremented) and the previous one is
    closed and fsynced. The header is written at file creation.
    """

    def __init__(self, cfg: WriterConfig) -> None:
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._file = None
        self._writer = None
        self._opened_at = 0.0
        self._lock = threading.Lock()
        self._fieldnames: Optional[list[str]] = cfg.fieldnames

    def _open_new(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
        path = self.cfg.output_dir / f"{self.cfg.base_name}_{self._seq:06d}.csv"
        self._file = path.open("w", newline="")
        # When proc_monitor runs under sudo (root), the CSV would otherwise
        # be owned by root with restrictive permissions, breaking analysis
        # done by the original user. We chmod 0644 (world-readable) and
        # chown back to the SUDO_USER when present. Failures are swallowed
        # because (1) when not running under sudo, no chown is needed and
        # (2) chmod on a file we just opened should always succeed for the
        # owning process.
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
        sudo_user = os.environ.get("SUDO_USER")
        if sudo_user:
            try:
                import pwd
                pwent = pwd.getpwnam(sudo_user)
                os.chown(str(path), pwent.pw_uid, pwent.pw_gid)
            except (KeyError, OSError):
                pass
        self._writer = None  # delayed until we know fieldnames
        self._opened_at = time.monotonic()
        self._seq += 1

    def write(self, sample: dict[str, Any]) -> None:
        with self._lock:
            now = time.monotonic()
            if self._file is None or (now - self._opened_at) >= self.cfg.rotation_seconds:
                self._open_new()
            if self._writer is None:
                fieldnames = self._fieldnames or sorted(sample.keys())
                self._fieldnames = fieldnames
                self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
                self._writer.writeheader()
            self._writer.writerow(sample)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None


class Watchdog:
    """Run a callable on a worker thread with a timeout.

    Returns the result if it completes in time, else returns the sentinel.
    The orphaned worker thread is left to finish on its own; we do not
    join it, since the whole point is that it might be deadlocked.
    """

    def __init__(self, timeout_seconds: float, sentinel: Any = None) -> None:
        self.timeout = timeout_seconds
        self.sentinel = sentinel

    def call(self, fn: Callable[[], Any]) -> Any:
        result_q: queue.Queue = queue.Queue(maxsize=1)

        def _runner() -> None:
            try:
                result_q.put(("ok", fn()))
            except BaseException as e:
                result_q.put(("err", e))

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            kind, payload = result_q.get(timeout=self.timeout)
        except queue.Empty:
            return self.sentinel
        if kind == "err":
            return self.sentinel
        return payload


class ShutdownEvent:
    """Unified shutdown signal handler."""

    def __init__(self) -> None:
        self._event = threading.Event()
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout=timeout)


def steady_sampler(
    sampler: Callable[[], dict[str, Any]],
    period_seconds: float,
    shutdown: ShutdownEvent,
    on_sample: Callable[[dict[str, Any]], None],
) -> None:
    """Drive a sampler at a steady period, robust to slow samples.

    Computes the next deadline as start + n*period to avoid drift. If a
    sample takes longer than period, deadlines are skipped (we do not
    backfill) and a warning is set in the sample as `_overrun=True`.
    """
    start = time.monotonic()
    n = 0
    while not shutdown.is_set():
        deadline = start + n * period_seconds
        now = time.monotonic()
        if now < deadline:
            shutdown.wait(deadline - now)
            if shutdown.is_set():
                break
        sample_started = time.monotonic()
        sample = sampler()
        sample_ended = time.monotonic()
        sample["_sample_duration_s"] = sample_ended - sample_started
        sample["_overrun"] = (sample_ended - deadline) > period_seconds
        sample["_wall_clock_unix"] = time.time()
        on_sample(sample)
        n += 1
