#!/usr/bin/env bash
# Build the training venv on a Leonardo LOGIN node (needs internet; compute nodes have none).
# Long pip installs may hit the 600 CPU-second login limit: run the heavy `pip install` lines
# from a data-mover / serial job if they are killed, or pre-download wheels with `pip download`.
#
#   bash research/slurm/setup_env.sh            # training venv
#   bash research/slurm/setup_env.sh --libero   # + LIBERO eval venv
#
# Stack (pinned, cu121 for the CUDA 12.2 driver):
#   python 3.11 (module cineca-ai binary or system python3.11), torch 2.5.1+cu121,
#   torchvision 0.20.1, flash-attn 2.7.4.post1 (prebuilt cu12/torch2.5 wheel), transformers
#   4.57.0, accelerate 1.5.2, deepspeed 0.16.9, torchdata 0.11, repo requirements.txt.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${HERE}/env.sh" || true
deactivate 2>/dev/null || true

PY=${PY:-python3.11}
if ! command -v "${PY}" >/dev/null 2>&1; then
  module load cineca-ai/4.3.0 2>/dev/null || true
  PY=$(command -v python3.11 || command -v python3)
  echo "using interpreter from module: ${PY}"
fi
PYPATH_SAVED=${PYTHONPATH:-}
unset PYTHONPATH   # do not inherit the module's site-packages into the venv

mkdir -p "$(dirname "${VENV}")"
if [ ! -x "${VENV}/bin/python" ]; then
  "${PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel setuptools==80.9.0

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# flash-attn prebuilt wheel (cu12, torch 2.5, cxx11abi FALSE matches the pip torch wheels)
FA_WHL="flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FA_WHL}" \
  || echo "!! flash-attn wheel install failed; set framework.qwenvl.attn_implementation=sdpa or build from source in a job"

# repo requirements minus the torchvision pin (0.21 pairs with torch 2.6; we use 0.20.1)
grep -v '^torchvision==' "${REPO_ROOT}/requirements.txt" > "${TMPDIR}/req.txt"
pip install -r "${TMPDIR}/req.txt"
pip install torchdata==0.11.0 pytest tensorboard

python - <<'EOF'
import torch, transformers, accelerate, deepspeed, torchdata
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "accelerate", accelerate.__version__, "deepspeed", deepspeed.__version__, "torchdata", torchdata.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn missing:", e)
EOF

if [ "${1:-}" = "--libero" ]; then
  deactivate
  if [ ! -x "${LIBERO_VENV}/bin/python" ]; then "${PY}" -m venv "${LIBERO_VENV}"; fi
  # shellcheck disable=SC1091
  source "${LIBERO_VENV}/bin/activate"
  python -m pip install -U pip wheel
  if [ ! -d "${LIBERO_HOME}" ]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_HOME}"; fi
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r "${LIBERO_HOME}/requirements.txt" || true
  pip install -e "${LIBERO_HOME}"
  pip install robosuite==1.4.1 "mujoco>=3.3" tyro websockets msgpack imageio[ffmpeg] opencv-python-headless matplotlib "numpy<2"
  # headless rendering on compute nodes
  echo 'export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl' > "${LIBERO_VENV}/egl.sh"
  python -c "import libero, robosuite, mujoco; print('LIBERO env ok', mujoco.__version__)"
fi
export PYTHONPATH="${PYPATH_SAVED}"
echo "done: VENV=${VENV} LIBERO_VENV=${LIBERO_VENV}"
