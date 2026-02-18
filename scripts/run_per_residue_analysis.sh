#!/bin/bash
#SBATCH --job-name=gef_per_res
#SBATCH --partition=dhvi-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/cwork/hsb26/GEF_inhibition/logs/per_res_%j.out
#SBATCH --error=/cwork/hsb26/GEF_inhibition/logs/per_res_%j.err

set -euo pipefail

# Keep everything off home dir
export TMPDIR="/scratch/hsb26/tmp"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"
export PIP_CACHE_DIR="/cwork/hsb26/pip_cache"
export XDG_CACHE_HOME="/cwork/hsb26/cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

echo "============================================================"
echo "Per-Residue Analysis (GPU)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Start:      $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

PYTHON="/cwork/hsb26/envs/gef/bin/python"
PROJECT="/cwork/hsb26/GEF_inhibition"

${PYTHON} -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

cd "${PROJECT}/src"
PYTHONUNBUFFERED=1 ${PYTHON} per_residue_analysis.py

echo ""
echo "============================================================"
echo "Complete: $(date)"
echo "============================================================"
