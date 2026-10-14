#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=belief
#SBATCH --output=logs/belief_%j.out
source lumi/common.sh
export PYTHONPATH=.
STEPS=${STEPS:-8000} B=128 EVAL_EVERY=1000 TAG=${TAG:-hasty} python train/train_belief.py
echo "=== BELIEF DONE"
