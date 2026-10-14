#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v2_train_%x_%j.out
# Track A v2 training from the current best net. Env: ARM R TRAIN (+ train_v2.py knobs)
source lumi/common.sh
export PYTHONPATH=.
INIT=$(cat data/v2_best_${ARM}.txt) TAG=v2_${ARM}_g$R python train_v2.py || exit 1
echo "=== V2 TRAIN DONE"
