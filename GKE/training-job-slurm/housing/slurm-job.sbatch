#!/usr/bin/env bash
#SBATCH --job-name=housing-price-train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/housing-%j.out
#SBATCH --error=/tmp/housing-%j.err

set -euo pipefail

: "${HOUSING_IMAGE:?HOUSING_IMAGE must be set (e.g. us-central1-docker.pkg.dev/<PROJECT_ID>/gke-training/housing-price-train:latest)}"
if [[ "$HOUSING_IMAGE" == *"YOUR_PROJECT_ID"* ]]; then
  echo "HOUSING_IMAGE still contains placeholder YOUR_PROJECT_ID; set a real project id" >&2
  exit 2
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=23456

export MASTER_ADDR MASTER_PORT
export WORLD_SIZE="$SLURM_NTASKS"

srun \
  --container-image="${HOUSING_IMAGE}" \
  bash -lc "torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_id=${SLURM_JOB_ID} /workspace/train.py"
