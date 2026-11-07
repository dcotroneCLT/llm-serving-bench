"""Shared run-integrity gate for short validation and production datasets.

Eligibility is independent of whether aging is significant. Stream coverage is
measured against the declared window, not just the rows that happened to arrive.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml


def finite(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric evidence")
    return value


def saved_client_config(run_dir: Path, manifest: dict) -> dict:
    config = dict((manifest.get("workload") or {}).get("client_config_overrides") or {})
    path = run_dir / "client_config.yaml"
    if path.exists():
        saved = yaml.safe_load(path.read_text())
        if not isinstance(saved, dict):
            raise ValueError("saved client config is not a mapping")
        config.update(saved)
    calibration = manifest.get("calibration") or {}
    if calibration.get("rate_calibrated_rps") is not None:
        config["target_rate_rps"] = calibration["rate_calibrated_rps"]
    return config


# started_at_unix is written before monitors and client are spawned, and the
# client counts duration_seconds from its own start (after process spawn and
# tokenizer load). The client does not record that start, so its submissions may
# legitimately land up to this many seconds past started_at + duration_s. Matches
# the 60 s cap of the monitor-stream grace in _stream_quality.
CLIENT_SPAWN_SLACK_S = 60.0


def _rows(run_dir: Path, prefix: str):
    for path in sorted(run_dir.glob(f"{prefix}_*.csv")):
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream)


def _stream_quality(run_dir, prefix, fields, start, duration, period, ended, alive=False):
    period = finite(period)
    if period <= 0:
        raise ValueError(f"{prefix}: invalid sample period")
    grace = max(3 * period, min(60, duration * 0.05))
    end = start + duration
    # Monitors keep sampling through the client drain (up to max(120, request_timeout + 90) s
    # past the declared window), so the upper bound is the manifest end, not start + duration.
    # Coverage below stays anchored on the declared window.
    hard_end = max(end, ended)
    previous = None
    count, first, last = 0, None, None
    for row in _rows(run_dir, prefix):
        stamp = finite(row["ts_unix"])
        if previous is not None and stamp <= previous:
            raise ValueError(f"{prefix}: duplicate/out-of-order timestamp")
        previous = stamp
        if stamp < start - grace or stamp > hard_end + grace:
            raise ValueError(f"{prefix}: sample outside declared run window")
        if alive and str(row.get("process_alive", "")).lower() not in ("true", "1"):
            continue
        try:
            populated = all(finite(row[field]) >= 0 for field in fields)
        except (ValueError, TypeError, KeyError):
            populated = False
        if not populated:
            continue
        if last is not None and stamp - last > max(60, 5 * period):
            raise ValueError(f"{prefix}: gap between valid samples exceeds continuity bound")
        first = stamp if first is None else first
        last = stamp
        count += 1
    expected = max(3, math.floor(0.95 * duration / period) - 1)
    if (count < expected or first is None or first > start + grace or last < end - grace):
        raise ValueError(f"{prefix}: insufficient valid coverage ({count} samples, need >= {expected}; "
                         f"first={first}, last={last}, target={start}..{end})")
    return f"    {prefix}: {count} valid samples cover the declared window"


def check_run_integrity(run_dir: Path, manifest: dict) -> tuple[bool, list[str]]:
    try:
        start = finite(manifest["started_at_unix"])
        ended = finite(manifest["ended_at_unix"])
        duration = finite(manifest["duration_s"])
        actual = finite(manifest["duration_seconds_actual"])
        if duration <= 0 or actual < 0.95 * duration or ended - start < 0.95 * duration:
            raise ValueError("run is not finalized at its declared duration")
        if (manifest.get("interrupted_early") is not False or
                manifest.get("client_forced_kill") or manifest.get("teardown_errors") or
                manifest.get("interruption_reason")):
            raise ValueError("run was interrupted, forcibly drained or failed teardown")
        monitors = manifest.get("monitors")
        if not isinstance(monitors, dict):
            raise ValueError("run has no recorded monitor configuration")
        component_spec = monitors.get("components") or {}
        prefix = manifest.get("proc_prefix") or (
            f"agg_{component_spec.get('engine_group', 'engine')}" if component_spec
            else (monitors.get("proc") or {}).get("label"))
        if not prefix:
            raise ValueError("missing canonical process series")
        messages = [_stream_quality(run_dir, prefix,
                    ["uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes"],
                    start, duration, monitors["proc"]["period_s"], ended, alive=True)]
        engine = manifest.get("engine") or {}
        devices = component_spec.get("gpu_devices") or manifest.get("gpu_devices") or engine.get("gpu_devices")
        if devices is None:
            topology = manifest.get("topology") or engine.get("topology") or {}
            devices = ([engine["gpu_device"]] if "gpu_device" in engine else
                       [topology[key] for key in ("prefill_gpu", "decode_gpu") if key in topology])
        if not devices:
            raise ValueError("missing monitored GPU device identities")
        for device in sorted(set(devices)):
            messages.append(_stream_quality(run_dir, f"gpu{int(device)}", ["vram_used_bytes"],
                            start, duration, monitors["gpu"]["period_s"], ended))
        messages.append(_stream_quality(run_dir, "system", ["mem_used_bytes"],
                        start, duration, monitors["system"]["period_s"], ended))
        config = saved_client_config(run_dir, manifest)
        rate = finite(config["target_rate_rps"])
        if rate <= 0:
            raise ValueError("missing/invalid recorded workload rate")
        client_grace = max(300, 10 / rate)
        first = last = None
        successful = 0
        for row in _rows(run_dir / "client", "requests"):
            stamp = finite(row["submitted_at_unix"])
            if stamp < start or stamp > start + duration + CLIENT_SPAWN_SLACK_S:
                raise ValueError("client submission outside declared run window")
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
            successful += row.get("status") == "ok"
        if (not successful or first is None or first > start + client_grace or
                last < start + duration - client_grace):
            raise ValueError("client stream is empty, all-failed or does not span the workload window")
        messages.append("    finalized run, populated process/GPU/system/client streams: OK")
        return True, messages
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return False, [f"    FAIL: run integrity: {exc}"]
