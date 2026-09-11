#!/bin/bash
#SBATCH --job-name=stage_libero
#SBATCH --account=AIFAC_P02_954
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --output=%x_%j.out
# LIBERO LeRobot datasets -> $SCRATCH/libero. lrd_all_serial is budget-free, 4 h max, and
# (unlike GPU nodes) can reach the internet through the login-side network. If downloads are
# refused here, run the same lines on a login node in chunks (600 CPU-s limit) or through the
# data mover (rclone/wget only).
set -euo pipefail
source "${WORK:-/leonardo_work/AIFAC_P02_954}/VLA-JEPA/research/slurm/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "${DATA}"

# four LIBERO suites in LeRobot format (mixture `libero_all` in gr00t_lerobot/mixtures.py)
for name in libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot \
            libero_spatial_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot; do
  echo "== ${name}"
  huggingface-cli download "IPEC-COMMUNITY/${name}" --repo-type dataset \
      --local-dir "${DATA}/${name}" --local-dir-use-symlinks False
  # modality.json expected by the GR00T LeRobot loader
  cp -n "${REPO_ROOT}/examples/LIBERO/modality.json" "${DATA}/${name}/meta/modality.json" || true
  echo "ok ${name}"
done
du -sh "${DATA}"/*
# keep $SCRATCH's 40-day idle cleanup away from the files
find "${DATA}" -type f -exec touch -a {} +
echo "staged"
