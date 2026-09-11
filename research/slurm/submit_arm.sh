#!/usr/bin/env bash
# Submit one arm: training job + dependent diagnostics + dependent LIBERO eval.
#   bash research/slurm/submit_arm.sh proj32
#   bash research/slurm/submit_arm.sh base_120k --nodes=2 --time=24:00:00
#   NO_EVAL=1 bash research/slurm/submit_arm.sh noz
# Requeues of the training job keep the same job id, so afterok dependencies survive restarts.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${HERE}/env.sh"
ARM=${1:?arm name}; shift || true
RUN_ID=${RUN_ID:-${ARM}}
[ -f "${REPO_ROOT}/scripts/configs/research/arms/${ARM}.yaml" ] || { echo "no arm ${ARM}"; exit 1; }

TRAIN_ID=$(ARM="${ARM}" RUN_ID="${RUN_ID}" sbatch --parsable --job-name="${RUN_ID}" "$@" "${HERE}/train.sbatch")
echo "train    ${RUN_ID}: ${TRAIN_ID}"
DIAG_ID=$(RUN_ID="${RUN_ID}" sbatch --parsable --job-name="diag_${RUN_ID}" --dependency=afterok:"${TRAIN_ID}" "${HERE}/diagnose.sbatch")
echo "diagnose ${RUN_ID}: ${DIAG_ID}"
if [ -z "${NO_EVAL:-}" ]; then
  EVAL_ID=$(RUN_ID="${RUN_ID}" sbatch --parsable --job-name="eval_${RUN_ID}" --dependency=afterok:"${TRAIN_ID}" "${HERE}/eval_libero.sbatch")
  echo "eval     ${RUN_ID}: ${EVAL_ID}"
fi
echo "${RUN_ID} train=${TRAIN_ID} diag=${DIAG_ID} eval=${EVAL_ID:-skipped} $(date -Is)" >> "${RUNS}/submissions.log"
