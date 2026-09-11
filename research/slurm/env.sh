# Leonardo environment for VLA-JEPA research jobs. Source from every sbatch / login shell.
#   source $WORK/VLA-JEPA/research/slurm/env.sh
# Verified 10 Sep 2026 (see research/LEONARDO_RUNBOOK.md): A100-SXM 64GB, driver 535 (CUDA
# 12.2) -> cu121 wheels; compute nodes have no internet; login nodes have a 600 CPU-s limit.

export PROJECT=AIFAC_P02_954
export WORK=${WORK:-/leonardo_work/${PROJECT}}
export FAST=${FAST:-/leonardo_scratch/fast/${PROJECT}}
export SCRATCH=${SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}

export REPO_ROOT=${REPO_ROOT:-${WORK}/VLA-JEPA}
export VENV=${VENV:-${FAST}/venv/vlajepa}
export LIBERO_VENV=${LIBERO_VENV:-${FAST}/venv/libero}
export LIBERO_HOME=${LIBERO_HOME:-${WORK}/LIBERO}
export MODELS=${FAST}/models
export DATA=${SCRATCH}/libero
export RUNS=${WORK}/runs
export CKPT_FAST=${FAST}/ckpt          # rolling full-state saves (regenerable)
export MILESTONES=${WORK}/milestones   # weights-only exports (must survive)

# caches: everything hot on $FAST, temp on $SCRATCH, no network from compute nodes
export HF_HOME=${FAST}/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${FAST}/torch_home
export TMPDIR=${SCRATCH}/tmp
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# NCCL over InfiniBand (4 HCAs per node); leave defaults unless a job hangs at init
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}" "${RUNS}" "${CKPT_FAST}" "${MILESTONES}" 2>/dev/null || true

module purge 2>/dev/null || true
module load profile/deeplrn 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

if [ -f "${VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
