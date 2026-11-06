# Scope audit corrections, 2026-11-06

Follow-up to commit 4d184d9. The original unit suite passed but omitted seven
cross-framework failure cases. The following corrections have dedicated
regressions in tests/test_scope_regressions.py, in addition to the existing
framework and statistical tests. Hardware checks remain pending.

Final local verification: all 524 framework tests plus 6 standalone stepness
tests passed under Linux/WSL. The generated-cell freshness check and git diff
whitespace check passed. No GPU/server run was launched by these tests.

| Finding | Correction | Regression evidence |
|---|---|---|
| PID1 GPU context swallowed all components | GPU coverage is separate from role assignment; inference requires a verified EngineCore subtree; unknown/mixed GPU roles fail explicitly | PID1 and EngineCore both on GPU retain distinct orchestration/engine/auxiliary totals; unclassified GPU workers fail |
| Process birth looked like mmap growth | Stepness loads PID/create-time membership and segments both RSS and VMS differences at boundaries | Persistent new helper with constant individual memory produces zero allocator steps; real batch CLI is exercised |
| Interrupted five-second run passed a 36-hour gate | Common finalization/status/duration/stream coverage gate precedes every layout; full analysis requires finite USS estimates and sufficient bins | Incomplete, interrupted, header-only, sparse, duplicate, all-failed and non-finite cases fail; a stationary complete run passes |
| Reparenting moved bytes between roles | Attribution is stable per process identity; ancestry changes are audited; validator rejects role transfer | Same helper reparented to PID1 retains its engine bytes; a fabricated transfer fails offline validation |
| Attach ignored incomplete/stale scope | Attach and managed lifecycle share ContainerScopeHealth and final flushed-evidence check | Actual mocked attach control flow fails alive-but-incomplete, alive-but-stale and empty final evidence, and accepts fresh complete evidence |
| Batch discovery skipped extension directories | Discover completed production DoW manifests using run_id/cell_id/finalization, preserving legacy naming | Batch CLI includes extension runs; live, interrupted, short validation and archived-copy directories are excluded |
| Feature gate assumed 0.8 prefix repeat | Resolve target from saved config, then manifest, with explicit CLI override; check zero targets and recorded arrival mode | Zero/0.4/0.8 shapes pass with the right target; any applied prefix at zero, missing flags or wrong arrival mode fail |

The scope schema is version 2. Its `gpu_pids` contains all measured GPU
contexts; `model_pids` is the subset assigned to the verified engine subtree.
Version-1 whole-container role results are not silently approved. This does not
alter the historical Dynamo PGID sampler or the retired single-process baseline.

Validation is about measurement eligibility, not finding a positive trend.
Short --scope-only checks still need to finish their declared window and
populate process/GPU/system/client streams. Full validation also needs usable
post-warmup statistics. Client drain and cleanup may extend wall-clock duration;
coverage is assessed on the declared measurement window rather than requiring
cleanup to finish within five percent of that window.

No production run data, calibration files, image pins, Dynamo deploy scripts,
workload design points or scheduling order are changed. The server remains
stopped pending the short-run checks, planned reboot, and Dynamo-only tail in
the operator runbook, docs/container_scope.md.
