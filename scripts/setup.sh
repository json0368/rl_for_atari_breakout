#!/usr/bin/env bash
# Create or update the project Conda environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
BREAKOUT_CONDA_ENV="${BREAKOUT_CONDA_ENV:-breakout}"
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    printf 'Conda not found at %s; set CONDA_ROOT to your Linux Conda directory.\n' "$CONDA_ROOT" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda run -n "$BREAKOUT_CONDA_ENV" python --version >/dev/null 2>&1; then
    conda env create -y -n "$BREAKOUT_CONDA_ENV" --solver libmamba -f configs/environment.yml
fi
conda activate "$BREAKOUT_CONDA_ENV"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]'
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable; check driver and GPU access"'
python -m pytest -q
printf '\nReady. Activate with: conda activate %s\n' "$BREAKOUT_CONDA_ENV"
