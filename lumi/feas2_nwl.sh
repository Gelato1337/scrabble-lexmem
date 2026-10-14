#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=100G
#SBATCH --time=06:00:00
#SBATCH --job-name=lex-feas2-nwl
#SBATCH --output=logs/feas2_nwl_%j.out
source lumi/common.sh
LEXICON=data/NWL23.txt TRAIN=hasty_train TEST=hasty_test STEPS=40000 B=64 NNEG=512 NPOS=256 EVAL_EVERY=4000 SLOT=0 TAG=nwl python feasibility/feas2_lexmem.py
echo "=== FEAS2 NWL DONE"
