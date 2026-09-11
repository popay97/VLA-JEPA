#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --account=AIFAC_P02_954
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# Latent-action diagnostics on one checkpoint (research/diagnose.py), 1 GPU, ~1-2 h.
#   RUN_ID=proj32 sbatch research/slurm/diagnose.sbatch
#   CKPT=$FAST/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt NAME=released_libero sbatch ...
set -euo pipefail
source "${WORK:-/leonardo_work/AIFAC_P02_954}/VLA-JEPA/research/slurm/env.sh"
cd "${REPO_ROOT}"
if [ -z "${CKPT:-}" ]; then
  RUN_ID=${RUN_ID:?set RUN_ID or CKPT}
  CKPT=$(ls -1 "${RUNS}/${RUN_ID}"/final_model/pytorch_model.pt 2>/dev/null || ls -1t "${RUNS}/${RUN_ID}"/checkpoints/steps_*_pytorch_model.pt | head -1)
  if [[ "${CKPT}" == */final_model/pytorch_model.pt ]]; then
    ln -sf "${CKPT}" "${RUNS}/${RUN_ID}/checkpoints/final_pytorch_model.pt"
    CKPT="${RUNS}/${RUN_ID}/checkpoints/final_pytorch_model.pt"
  fi
fi
NAME=${NAME:-${RUN_ID:-ckpt}}
OUT=${WORK}/results/${NAME}/diagnostics
mkdir -p "${OUT}"
python research/diagnose.py --ckpt "${CKPT}" --out "${OUT}" \
  --data_root "${DATA}" --data_mix libero_all --num_batches "${NUM_BATCHES:-64}" --batch_size 8 \
  --base_vlm "${MODELS}/Qwen3-VL-2B-Instruct" ${DIAG_ARGS:-} 2>&1 | tee "${OUT}/diagnose.log"
echo "[$(date)] diagnostics done: ${OUT}"
