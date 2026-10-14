#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/td_gen_%x_%A_%a.out
# Track A self-play generation, one node per array task. Env: ARM R CKPT ACT NGT (games per task) SEEDBASE
source lumi/common.sh
export PYTHONPATH=.
I=${SLURM_ARRAY_TASK_ID:-0}
CKPT=$CKPT ACT=$ACT NPROC=${NPROC:-124} TAG=td_${ARM}_g${R}_$I python selfplay/selfplay_td.py $NGT $((SEEDBASE + I * 100000)) || exit 1
echo "=== GEN DONE"
