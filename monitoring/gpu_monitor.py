"""GPU monitor: samples NVML metrics for one GPU at a fixed rate.

Default rate is 1 Hz. Sampled fields:

  ts_unix           wall-clock timestamp at the start of the sample
  gpu_index         the GPU device index (passed via --gpu-index)
  vram_used_bytes
  vram_free_bytes
  vram_total_bytes
  gpu_util_percent  SM utilization
  mem_util_percent  memory controller utilization (separate from VRAM used)
  temperature_c
  power_draw_w
  power_limit_w
  sm_clock_mhz
  mem_clock_mhz
  pstate
  ecc_db_volatile   double-bit ECC errors, volatile counter
  ecc_sb_volatile   single-bit ECC errors, volatile counter
  fan_percent       0 if no fan / data-center cards
  pcie_tx_kb        per-second throughput estimate (delta / period)
  pcie_rx_kb
  _sample_duration_s
  _overrun
  _wall_clock_unix

Some fields may be absent on certain cards. Missing values are written
as empty strings in the CSV.

Watchdog wraps the NVML query with a 30 s timeout. If NVML deadlocks,
the sample is recorded as a sentinel row (all measured fields empty,
_overrun=True) so the time series remains aligned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import pynvml
except ImportError as e:
    raise SystemExit("pynvml not installed. Run: pip install nvidia-ml-py") from e

from _common import (
    CsvRotatingWriter,
    ShutdownEvent,
    Watchdog,
    WriterConfig,
    steady_sampler,
)


FIELDNAMES = [
    "ts_unix",
    "gpu_index",
    "vram_used_bytes",
    "vram_free_bytes",
    "vram_total_bytes",
    "gpu_util_percent",
    "mem_util_percent",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "pstate",
    "ecc_db_volatile",
    "ecc_sb_volatile",
    "fan_percent",
    "pcie_tx_kb",
    "pcie_rx_kb",
    "_sample_duration_s",
    "_overrun",
    "_wall_clock_unix",
]


def _safe(call):
    try:
        return call()
    except pynvml.NVMLError:
        return None


def make_sampler(handle, gpu_index: int):
    def sample() -> dict[str, Any]:
        import time

        ts = time.time()
        mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
        temp = _safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        power_limit = _safe(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
        sm_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        mem_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        pstate = _safe(lambda: pynvml.nvmlDeviceGetPerformanceState(handle))
        ecc_db = _safe(
            lambda: pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC
            )
        )
        ecc_sb = _safe(
            lambda: pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_VOLATILE_ECC
            )
        )
        fan = _safe(lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
        pcie_tx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
        pcie_rx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))

        return {
            "ts_unix": ts,
            "gpu_index": gpu_index,
            "vram_used_bytes": mem.used if mem else None,
            "vram_free_bytes": mem.free if mem else None,
            "vram_total_bytes": mem.total if mem else None,
            "gpu_util_percent": util.gpu if util else None,
            "mem_util_percent": util.memory if util else None,
            "temperature_c": temp,
            "power_draw_w": power,
            "power_limit_w": power_limit,
            "sm_clock_mhz": sm_clock,
            "mem_clock_mhz": mem_clock,
            "pstate": pstate,
            "ecc_db_volatile": ecc_db,
            "ecc_sb_volatile": ecc_sb,
            "fan_percent": fan,
            "pcie_tx_kb": pcie_tx,
            "pcie_rx_kb": pcie_rx,
        }

    return sample


def main() -> None:
    p = argparse.ArgumentParser(description="GPU NVML monitor.")
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--period-seconds", type=float, default=1.0)
    p.add_argument("--rotation-seconds", type=int, default=60)
    p.add_argument("--watchdog-timeout-s", type=float, default=30.0)
    args = p.parse_args()

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        sampler = make_sampler(handle, args.gpu_index)
        watchdog = Watchdog(timeout_seconds=args.watchdog_timeout_s, sentinel={"gpu_index": args.gpu_index})
        writer = CsvRotatingWriter(
            WriterConfig(
                output_dir=args.output_dir,
                base_name=f"gpu{args.gpu_index}",
                rotation_seconds=args.rotation_seconds,
                fieldnames=FIELDNAMES,
            )
        )
        shutdown = ShutdownEvent()

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
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
