#!/usr/bin/env bash
# Pin engine container images to wosar2026_* local tags and record their
# digests in engines/<name>/image_pin_<id>.json so subsequent campaign runs
# use a frozen-in-time image regardless of whether the upstream tag is
# repushed by Anthropic/NVIDIA/vLLM after the campaign starts.
#
# Run once on cci-csgpu11 before any cell yaml references the pin files.
# Re-running is safe and idempotent.
#
# Outputs (one file per distinct image, NOT per engine directory, to
# avoid clobbering when one directory pins multiple images):
#   engines/vllm_standalone/image_pin_e1.json    (vLLM V1, latest)
#   engines/vllm_standalone/image_pin_a1.json    (vLLM V0, v0.7.3)
#   engines/triton_vllm/image_pin.json           (shared E2 and A2)
#   engines/pytorch_naive/image_pin.json         (locally built)
#
# image_pin_*.json schema:
#   { "image_tag":  "<local tag>",
#     "image_repo": "<source repo>",
#     "source_tag": "<original upstream tag>",
#     "digest":     "sha256:...",
#     "pinned_at":  "<UTC ISO timestamp>" }

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/wosar/llm-serving-bench}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "[pin_images] REPO_ROOT not found: $REPO_ROOT" >&2
  exit 1
fi

# pin_image SOURCE_REF LOCAL_TAG ENGINE_DIR PIN_FILENAME
#
# SOURCE_REF can be either a remote pullable tag (vllm/vllm-openai:latest)
# or a local-only tag (pytorch_naive:wosar2026). The function:
#   1. Skips `docker pull` if the image is already present locally
#      (avoids the "pull access denied" failure on local-only refs).
#   2. Tags SOURCE_REF as LOCAL_TAG.
#   3. Writes the pin file at ENGINE_DIR/PIN_FILENAME.
pin_image() {
  local source_ref="$1"
  local local_tag="$2"
  local engine_dir="$3"
  local pin_filename="$4"

  if docker image inspect "$source_ref" >/dev/null 2>&1; then
    echo "[pin_images] $source_ref already present locally, skipping pull"
  else
    echo "[pin_images] pulling $source_ref"
    docker pull "$source_ref"
  fi

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
  local pin_file="$pin_dir/$pin_filename"
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

# E1: vLLM standalone V1 (latest at pin time, default scheduler V1).
pin_image \
  "vllm/vllm-openai:latest" \
  "vllm/vllm-openai:wosar2026_e1" \
  "engines/vllm_standalone" \
  "image_pin_e1.json"

# A1: vLLM standalone V0 (pinned to v0.7.3, last release where V0 was default).
pin_image \
  "vllm/vllm-openai:v0.7.3" \
  "vllm/vllm-openai:wosar2026_a1" \
  "engines/vllm_standalone" \
  "image_pin_a1.json"

# E2 + A2: Triton + vLLM. Same image, env vars differ between cells.
pin_image \
  "nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3" \
  "tritonserver:wosar2026_e2_a2" \
  "engines/triton_vllm" \
  "image_pin.json"

# E3 + E3b: PyTorch naive, locally built. Build if not yet present.
if ! docker image inspect "pytorch_naive:wosar2026" >/dev/null 2>&1; then
  echo "[pin_images] building pytorch_naive:wosar2026 from $REPO_ROOT/engines/pytorch_naive"
  docker build -t "pytorch_naive:wosar2026" "$REPO_ROOT/engines/pytorch_naive"
fi
pin_image \
  "pytorch_naive:wosar2026" \
  "pytorch_naive:wosar2026" \
  "engines/pytorch_naive" \
  "image_pin.json"

echo ""
echo "[pin_images] done. Pinned tags and digests:"
for pin in \
    "$REPO_ROOT/engines/vllm_standalone/image_pin_e1.json" \
    "$REPO_ROOT/engines/vllm_standalone/image_pin_a1.json" \
    "$REPO_ROOT/engines/triton_vllm/image_pin.json" \
    "$REPO_ROOT/engines/pytorch_naive/image_pin.json"; do
  if [ -f "$pin" ]; then
    echo "  $pin"
    cat "$pin" | sed 's/^/    /'
  fi
done

echo ""
echo "[pin_images] docker images with wosar2026 tag:"
docker images --digests | grep -E "wosar2026|REPOSITORY" || true
