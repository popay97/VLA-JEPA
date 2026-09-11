#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --account=AIFAC_P02_954
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# LIBERO eval (4 suites x 10 tasks x 50 trials) of one checkpoint on one GPU: policy server
# (training venv) + eval client (LIBERO venv, EGL rendering), suites run sequentially on the
# same GPU to fit the 1-GPU/1-slot budget. ~10 GPU-h.
#   RUN_ID=proj32 sbatch research/slurm/eval_libero.sbatch                 # final_model
#   CKPT=$FAST/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt sbatch ...  # released
set -euo pipefail
source "${WORK:-/leonardo_work/AIFAC_P02_954}/VLA-JEPA/research/slurm/env.sh"
cd "${REPO_ROOT}"

if [ -z "${CKPT:-}" ]; then
  RUN_ID=${RUN_ID:?set RUN_ID or CKPT}
  CKPT=$(ls -1 "${RUNS}/${RUN_ID}"/final_model/pytorch_model.pt 2>/dev/null || ls -1t "${RUNS}/${RUN_ID}"/checkpoints/steps_*_pytorch_model.pt | head -1)
fi
NAME=${RUN_ID:-$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)}')}
OUT=${WORK}/results/${NAME}/libero
mkdir -p "${OUT}"
echo "[$(date)] eval ${CKPT} -> ${OUT}"

# read_mode_config expects <run>/checkpoints/<ckpt>.pt with <run>/config.yaml and
# dataset_statistics.json; final_model/pytorch_model.pt is one level up, so mirror it
if [[ "${CKPT}" == */final_model/pytorch_model.pt ]]; then
  RUN_DIR=$(dirname "$(dirname "${CKPT}")")
  ln -sf "${CKPT}" "${RUN_DIR}/checkpoints/final_pytorch_model.pt"
  CKPT="${RUN_DIR}/checkpoints/final_pytorch_model.pt"
fi

PORT=$((15000 + SLURM_JOB_ID % 10000))
python deployment/model_server/server_policy.py --ckpt_path "${CKPT}" --port "${PORT}" --use_bf16 --cuda 0 \
  > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 120); do grep -q "server running" "${OUT}/server.log" 2>/dev/null && break; sleep 5; done
grep -q "server running" "${OUT}/server.log" || { echo "server failed"; cat "${OUT}/server.log"; exit 1; }

export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
SIM_PY=${LIBERO_VENV}/bin/python
TRIALS=${TRIALS:-50}
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "[$(date)] suite ${suite}"
  mkdir -p "${OUT}/${suite}"
  "${SIM_PY}" examples/LIBERO/eval_libero.py \
    --args.pretrained-path "${CKPT}" --args.host 127.0.0.1 --args.port "${PORT}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task "${TRIALS}" \
    --args.video-out-path "${OUT}/${suite}" --args.with_state true \
    > "${OUT}/${suite}/eval.log" 2>&1 || echo "suite ${suite} failed (see eval.log)"
  grep -iE "success|rate" "${OUT}/${suite}/eval.log" | tail -3 || true
done
python research/collect_results.py --results "${WORK}/results" --out "${WORK}/results/summary.md" || true
echo "[$(date)] eval done: ${OUT}"
