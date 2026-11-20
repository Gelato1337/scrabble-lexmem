#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/value_leaf_%j.out
# Our own search leaf: train/value_leaf.py (env DATA INIT USE_LEX RETR STEPS B LR TAG via --export=ALL); PRE: shell command run
# first (e.g. selfplay/ei_data.py values).
source lumi/common.sh
export PYTHONPATH=.
[ -n "$PRE" ] && { eval "$PRE" || exit 1; }
python train/value_leaf.py || exit 1
echo "=== VALUE LEAF DONE"
