#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=08:00:00
#SBATCH --job-name=train-value
#SBATCH --output=logs/value_%x_%j.out
source lumi/common.sh
export PYTHONPATH=.
# USE_LEX, TAG, RETR, LABELS via --export
LABELS=${LABELS:-data/simlabels_hasty300k.pkl} STEPS=${STEPS:-10000} B=64 EVAL_EVERY=1000 python train_value.py
echo "=== VALUE DONE"
