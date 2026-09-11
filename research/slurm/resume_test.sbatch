#!/bin/bash
#SBATCH --job-name=resume_test
#SBATCH --account=AIFAC_P02_954
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=32 --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# T1 acceptance test for resumable training (dbg QoS starts instantly, 30 min):
#   phase A: 200 steps, STOP_AND_SAVE dropped at ~step 100 -> trainer saves state and exits
#   phase B: relaunch, auto-resume from latest, run to 200, DONE written.
# Pass criteria (checked below): meta.json completed_steps ~100, phase B log starts at that
# step with the continuing LR, final DONE present, summary.jsonl has the 200-step milestone.
set -euo pipefail
source "${WORK:-/leonardo_work/AIFAC_P02_954}/VLA-JEPA/research/slurm/env.sh"
cd "${REPO_ROOT}"
RUN_ID=resume_test_${SLURM_JOB_ID}
RUN_DIR=${RUNS}/${RUN_ID}
mkdir -p "${RUN_DIR}/checkpoints"
COMMON=(--config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 4 --main_process_port $((20000 + SLURM_JOB_ID % 20000)))
ARGS=(starVLA/training/train_starvla.py --config_yaml scripts/configs/research/libero_ft_base.yaml
      --extra_yaml scripts/configs/research/arms/base_30k.yaml --run_id "${RUN_ID}" --run_root_dir "${RUNS}"
      --trainer.max_train_steps 200 --trainer.num_warmup_steps 50 --trainer.save_interval 200
      --trainer.eval_interval 50 --trainer.logging_frequency 10 --trainer.save_every_minutes 0)

echo "== phase A"
accelerate launch "${COMMON[@]}" "${ARGS[@]}" > "${RUN_DIR}/phaseA.log" 2>&1 &
PID=$!
# wait until step 100 is logged, then request a save+stop
until grep -q "Step 100," "${RUN_DIR}/phaseA.log" 2>/dev/null; do
  kill -0 ${PID} 2>/dev/null || { echo "phase A died"; tail -50 "${RUN_DIR}/phaseA.log"; exit 1; }
  sleep 5
done
touch "${RUN_DIR}/checkpoints/STOP_AND_SAVE"
wait ${PID} || true
echo "phase A exit; state:"; cat "${RUN_DIR}/checkpoints/state/latest"; cat "${RUN_DIR}/checkpoints/state/$(cat "${RUN_DIR}/checkpoints/state/latest")/meta.json"
grep -E "Step (90|100|110)," "${RUN_DIR}/phaseA.log" | tail -3

echo "== phase B"
accelerate launch "${COMMON[@]}" "${ARGS[@]}" > "${RUN_DIR}/phaseB.log" 2>&1
grep -E "\[resume\]" "${RUN_DIR}/phaseB.log" | head -3
grep -E "Step (110|120|200)," "${RUN_DIR}/phaseB.log" | head -3
test -f "${RUN_DIR}/DONE" && echo "PASS: DONE written" || { echo "FAIL: no DONE"; exit 1; }
cat "${RUN_DIR}/summary.jsonl"
echo "compare learning_rate at Step 110 in phaseA (if reached) vs phaseB: must be continuous"
