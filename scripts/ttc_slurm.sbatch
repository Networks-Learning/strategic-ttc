#!/bin/bash

#SBATCH -J strategic_ttc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=23:59:00
#SBATCH --partition=a100,a40
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH -o logs/ttc_%j.out
#SBATCH -e logs/ttc_%j.err

# Usage:
#   sbatch scripts/ttc_slurm.sbatch <CONFIG>
#

set -euo pipefail

CONFIG="${1:?ERROR: missing CONFIG file path}"

echo "[strategic-ttc slurm] CONFIG: $CONFIG"
echo "[strategic-ttc slurm] NODE: $(hostname)"
echo "[strategic-ttc slurm] GPU: $CUDA_VISIBLE_DEVICES"

# Activate virtual environment
source ../svenv/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

strategic-ttc-generate \
    -c "$CONFIG" \

deactivate
echo "[strategic-ttc slurm] Done."
