#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60G
#SBATCH --time=10:00:00
#SBATCH --job-name=lex-feas2
#SBATCH --output=logs/feas2_%x_%j.out
source lumi/common.sh
# SLOT and TAG come from --export on submission
TRAIN=big TEST=bigtest STEPS=40000 B=64 NNEG=512 NPOS=256 EVAL_EVERY=2000 python feas2_lexmem.py
echo "=== FEAS2 DONE"
