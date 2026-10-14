#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/td_train_%x_%j.out
# Track A TD training. Env: TRAIN INIT TAG LAM STEPS LR
source lumi/common.sh
export PYTHONPATH=.
python train/train_td.py || exit 1
echo "=== TD TRAIN DONE"
