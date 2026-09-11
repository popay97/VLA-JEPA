#!/usr/bin/env bash
# Stage base models and released checkpoints into $FAST/models. Run on a LOGIN node (internet).
# Each download is resumable; re-run until every `ok` line prints. Uses huggingface_hub from
# the training venv (pip install "huggingface_hub[cli]" if missing).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${HERE}/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "${MODELS}"

dl() {  # repo_id  target_dir  [extra hf args]
  local repo=$1 dst=$2; shift 2
  echo "== ${repo} -> ${dst}"
  huggingface-cli download "${repo}" --local-dir "${dst}" --local-dir-use-symlinks False "$@"
  echo "ok ${repo}"
}

dl Qwen/Qwen3-VL-2B-Instruct            "${MODELS}/Qwen3-VL-2B-Instruct"
dl facebook/vjepa2-vitl-fpc64-256        "${MODELS}/vjepa2-vitl-fpc64-256"
dl galilai-group/LeVJEPA-VideoMix-Large  "${MODELS}/LeVJEPA-VideoMix-Large"
dl ginwind/VLA-JEPA                      "${MODELS}/VLA-JEPA" --include "Pretrain/*" "LIBERO/*"

du -sh "${MODELS}"/*
echo "models staged under ${MODELS}"
