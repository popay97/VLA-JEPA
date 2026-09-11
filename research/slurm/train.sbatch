#!/bin/bash
#SBATCH --job-name=vlajepa
#SBATCH --account=AIFAC_P02_954
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --output=%x_%j.out
#
# One fine-tune arm on one node (4 x A100 64GB). Requeue chain: 15 min before the limit Slurm
# sends USR1 to this batch shell; the trap touches STOP_AND_SAVE, the trainer saves its full
# state and exits, and the script requeues itself unless <run>/DONE exists. Every restart
# auto-resumes from checkpoints/state/latest.
#
# Usage (normally through submit_arm.sh):
#   ARM=proj32 sbatch --job-name=proj32 research/slurm/train.sbatch
#   ARM=base_120k NGPU=4 sbatch --nodes=2 --job-name=base_120k research/slurm/train.sbatch
# Extra CLI overrides: EXTRA_ARGS="--trainer.max_train_steps 200 --trainer.save_interval 100"
set -euo pipefail
source "${WORK:-/leonardo_work/AIFAC_P02_954}/VLA-JEPA/research/slurm/env.sh"

ARM=${ARM:?set ARM=<arm name under scripts/configs/research/arms>}
BASE_YAML=${BASE_YAML:-${REPO_ROOT}/scripts/configs/research/libero_ft_base.yaml}
ARM_YAML=${ARM_YAML:-${REPO_ROOT}/scripts/configs/research/arms/${ARM}.yaml}
RUN_ID=${RUN_ID:-${ARM}}
RUN_DIR=${RUNS}/${RUN_ID}
NGPU=${NGPU:-4}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NPROC=$((NGPU * NNODES))
DS_YAML=${DS_YAML:-${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs"
cd "${REPO_ROOT}"
echo "[$(date)] job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} arm=${ARM} run=${RUN_DIR} nodes=${NNODES} procs=${NPROC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
git -C "${REPO_ROOT}" rev-parse --short HEAD > "${RUN_DIR}/logs/commit_${SLURM_JOB_ID}.txt" || true

if [ -f "${RUN_DIR}/DONE" ]; then echo "already DONE"; exit 0; fi
rm -f "${RUN_DIR}/checkpoints/STOP_AND_SAVE"

on_usr1() {
  echo "[$(date)] USR1: asking trainer to save and stop"
  touch "${RUN_DIR}/checkpoints/STOP_AND_SAVE"
}
trap on_usr1 USR1

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

LAUNCH_ARGS=(--config_file "${DS_YAML}" --num_processes "${NPROC}" --num_machines "${NNODES}" --main_process_port "${MASTER_PORT}")
if [ "${NNODES}" -gt 1 ]; then
  LAUNCH_ARGS+=(--main_process_ip "${MASTER_ADDR}" --machine_rank "\${SLURM_NODEID}")
  # one accelerate launcher per node
  srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c \
    "source ${REPO_ROOT}/research/slurm/env.sh && accelerate launch ${LAUNCH_ARGS[*]} \
       ${REPO_ROOT}/starVLA/training/train_starvla.py --config_yaml ${BASE_YAML} --extra_yaml ${ARM_YAML} \
       --run_id ${RUN_ID} --run_root_dir ${RUNS} ${EXTRA_ARGS}" \
    > "${RUN_DIR}/logs/train_${SLURM_JOB_ID}.log" 2>&1 &
else
  accelerate launch "${LAUNCH_ARGS[@]}" \
    "${REPO_ROOT}/starVLA/training/train_starvla.py" --config_yaml "${BASE_YAML}" --extra_yaml "${ARM_YAML}" \
    --run_id "${RUN_ID}" --run_root_dir "${RUNS}" ${EXTRA_ARGS} \
    > "${RUN_DIR}/logs/train_${SLURM_JOB_ID}.log" 2>&1 &
fi
TRAIN_PID=$!
# `wait` returns when the signal arrives; loop until the trainer really exits
set +e
while kill -0 "${TRAIN_PID}" 2>/dev/null; do wait "${TRAIN_PID}"; done
RC=$?
set -e
echo "[$(date)] trainer exited rc=${RC}"

if [ -f "${RUN_DIR}/DONE" ]; then
  echo "run complete"
  if [ -n "${CHAIN_AFTER:-}" ]; then bash "${CHAIN_AFTER}" "${RUN_ID}" || true; fi
  exit 0
fi
if [ -f "${RUN_DIR}/checkpoints/state/latest" ]; then
  echo "[$(date)] not done, requeueing ${SLURM_JOB_ID} (resume from $(cat "${RUN_DIR}/checkpoints/state/latest"))"
  scontrol requeue "${SLURM_JOB_ID}" || echo "requeue refused (ask CINECA for requeue permission); resubmit manually"
  exit 0
fi
echo "no state to resume from; exiting with rc=${RC}"
exit "${RC}"
