#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60G
#SBATCH --time=10:00:00
#SBATCH --job-name=lex-feas4
#SBATCH --output=logs/feas4_%j.out
source lumi/common.sh
TRAIN=big TEST=bigtest STEPS=30000 B=48 K=4 EVAL_EVERY=2000 TAG=latent INIT=data/lexmem_base.pt python feasibility/feas4_latent.py
echo "=== FEAS4 DONE"
