# HARD CONSTRAINT — cross-system vLLM pin (extension campaign)

Status: locked 2026-06-29. Applies to the whole extension campaign (Dynamo +
Triton+vLLM + standalone vLLM). This is the campaign's core methodological
improvement over the preprint; treat it as a gate, not a guideline.

## The rule

The three serving systems **must run the IDENTICAL vLLM version**, pinned by
**image digest**, on every run of the campaign:

- **Dynamo** (disaggregated/aggregated)
- **Triton + vLLM backend**
- **vLLM standalone**

The point of the extension is to compare the *hosting/orchestration layer* with
the inference engine held constant. The preprint was confounded by vLLM version
drift across cells; repeating that would invalidate the comparison. So a run is
only valid if all three carry the same vLLM.

## What counts as a valid pin

The pinned version must be one that **all three ship NATIVELY** — i.e. it lies in
the intersection:

```
{ vLLM in a STABLE Dynamo release } ∩ { vLLM in a Triton release } ∩ { vllm-openai tag }
```

No forced/custom installs (e.g. pip-installing a different vLLM into the Triton
backend) — that reintroduces a per-system build difference and defeats the point.
If the intersection is empty at a desired version, move the whole pin to a
version that is in the intersection.

## Authority: `pip show vllm` on the box, nothing else

Remote sources (NVIDIA release notes, `server/build.py` version_map, blog posts)
are **NOT authoritative** and have been wrong: `build.py` r26.03 claimed vLLM
0.16.0 while the actual container shipped **0.17.1**. The ONLY authority is
`pip show vllm` run inside each pulled image. Verify all three before any run;
**STOP** if any is not the pinned version.

```bash
docker run --rm <dynamo_img>   pip show vllm | grep -i '^Version'
docker run --rm <triton_img>   pip show vllm | grep -i '^Version'
docker run --rm --entrypoint pip <vllm_openai_img> show vllm | grep -i '^Version'   # entrypoint = api server
```

(The NVIDIA *vLLM container* release notes did match ground truth on 26.03=0.17.1,
so they are a reasonable way to NARROW candidates — but the box `pip show` is the
final word.)

## Current pin (locked 2026-06-29): vLLM 0.20.1

| System | Image | vLLM |
|---|---|---|
| Dynamo | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13` | 0.20.1 |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3` | 0.20.1 |
| Standalone | `vllm/vllm-openai:v0.20.1-cu129` | 0.20.1 |

Digests are recorded in `engines/{dynamo_vllm,triton_vllm,vllm_standalone}/image_pin*.json`
at pull time (`docker inspect --format '{{index .RepoDigests 0}}'`).

History: 0.16.0 was the first pick (Dynamo 1.0.1) but the box gate found Triton
ships **no** 0.16.0 build, so the intersection was empty there; re-derived to
0.20.1.

## Version map (for the next re-pin)

Ground-truthed / from the reliable vLLM-container notes:

| Triton / vLLM container | vLLM | | Dynamo stable | vLLM |
|---|---|---|---|---|
| 26.01 | 0.11.1 | | 1.0.1 | 0.16.0 (no Triton match) |
| 26.02 | 0.15.1 | | 1.1.1 | 0.19.0 (Triton 26.04) |
| 26.03 | 0.17.1 | | 1.2.0 | 0.20.1 (Triton 26.05) |
| 26.04 | 0.19.0 | | | |
| 26.05 | 0.20.1 | | | |

Clean intersections: **0.19.0** (Dynamo 1.1.1 + Triton 26.04) and **0.20.1**
(Dynamo 1.2.0 + Triton 26.05). 0.16.0 and 0.17.1 have no stable three-way match.

## Corollaries

1. **Never bump one system's release alone.** To change vLLM, move all three to
   the same new version from the intersection, then re-run the `pip show` gate.
2. **Per-image substrate differs and is declared in Threats to Validity.** Each
   image carries its own CUDA / torch / glibc / allocator — precisely what the
   stepness metric probes. This is intrinsic to any cross-system comparison (you
   cannot run "Triton" without Triton's image) and is mitigated by holding the
   vLLM orchestration layer identical. It is strictly better than the preprint's
   vLLM version-drift confound; do not paper over it, declare it.
3. **CUDA vs host driver.** Dynamo/Triton are CUDA 13.x; the standalone
   `vllm/vllm-openai:v0.20.1-cu129` is CUDA 12.9 (upstream 0.20.1 ships no
   `-cu130`). Both run on the host driver 580.159.03 (supports CUDA 13.x, and
   12.9 backward-compat). The CUDA delta is part of the per-image substrate
   caveat (#2); vLLM 0.20.1 is identical across all three.
