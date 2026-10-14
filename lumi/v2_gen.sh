#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2_gen_%x_%A_%a.out
# Track A v2 generation, one node per array task. The current best net is read at run time (gating). Env: ARM R NGT SEEDBASE
source lumi/common.sh
export PYTHONPATH=.
I=${SLURM_ARRAY_TASK_ID:-0}; CKPT=$(cat data/v2_best_${ARM}.txt)
echo "generator net: $CKPT"
CKPT=$CKPT NPROC=${NPROC:-124} TAG=v2_${ARM}_g${R}_$I python selfplay_v2.py $NGT $((SEEDBASE + I * 100000)) || exit 1
echo "=== GEN DONE"
