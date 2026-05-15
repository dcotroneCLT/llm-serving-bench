#!/usr/bin/env bash
# Pin engine container images to wosar2026_* local tags and record their
# digests in engines/<name>/image_pin.json so subsequent campaign runs
# use a frozen-in-time image regardless of whether the upstream tag is
# repushed by Anthropic/NVIDIA/vLLM after the campaign starts.
#
# Run once on cci-csgpu11 before pin_images.sh is referenced by any
# cell yaml. Re-running is safe and idempotent.
#
# Outputs:
#   engines/vllm_standalone/image_pin.json
#   engines/triton_vllm/image_pin.json
#   engines/pytorch_naive/image_pin.json
#
# image_pin.json schema:
#   { "image_tag": "<local tag>",
#     "image_repo": "<source repo>",
#     "source_tag": "<original upstream tag>",
#     "digest": "sha256:...",
#     "pinned_at": "<UTC ISO timestamp>" }

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/wosar/llm-serving-bench}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "[pin_images] REPO_ROOT not found: $REPO_ROOT" >&2
  exit 1
fi

pin_image() {
  local source_ref="$1"     # e.g. vllm/vllm-openai:latest
  local local_tag="$2"      # e.g. vllm/vllm-openai:wosar2026_e1
  local engine_dir="$3"     # e.g. engines/vllm_standalone

  echo "[pin_images] pulling $source_ref"
  docker pull "$source_ref"

  local digest
  digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$source_ref" 2>/dev/null | sed 's/.*@//')
  if [ -z "$digest" ]; then
    # Locally built image: no RepoDigest. Use image ID prefixed with sha256:.
    digest=$(docker inspect --format '{{.Id}}' "$source_ref")
  fi

  echo "[pin_images] tagging $source_ref -> $local_tag"
  docker tag "$source_ref" "$local_tag"

  local pin_dir="$REPO_ROOT/$engine_dir"
  mkdir -p "$pin_dir"
  local pin_file="$pin_dir/image_pin.json"
  cat > "$pin_file" <<EOF
{
  "image_tag": "$local_tag",
  "image_repo": "$(echo "$source_ref" | cut -d: -f1)",
  "source_tag": "$source_ref",
  "digest": "$digest",
  "pinned_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  echo "[pin_images] wrote $pin_file"
}

# vLLM standalone, V1 (latest at pin time, default scheduler V1).
pin_image \
  "vllm/vllm-openai:latest" \
  "vllm/vllm-openai:wosar2026_e1" \
  "engines/vllm_standalone"

# vLLM standalone, V0 (pinned to v0.7.3, last release where V0 was default).
pin_image \
  "vllm/vllm-openai:v0.7.3" \
  "vllm/vllm-openai:wosar2026_a1" \
  "engines/vllm_standalone"

# Triton + vLLM. Same image serves both E2 (V0 default) and A2 (V1 forced via env).
pin_image \
  "nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3" \
  "tritonserver:wosar2026_e2_a2" \
  "engines/triton_vllm"

# PyTorch naive: locally built. If the image does not exist yet, build it.
if ! docker image inspect "pytorch_naive:wosar2026" >/dev/null 2>&1; then
  echo "[pin_images] building pytorch_naive:wosar2026 from $REPO_ROOT/engines/pytorch_naive"
  docker build -t "pytorch_naive:wosar2026" "$REPO_ROOT/engines/pytorch_naive"
fi
pin_image \
  "pytorch_naive:wosar2026" \
  "pytorch_naive:wosar2026" \
  "engines/pytorch_naive"

echo ""
echo "[pin_images] done. Pinned tags:"
docker images --digests | grep -E "wosar2026|REPOSITORY" || true
