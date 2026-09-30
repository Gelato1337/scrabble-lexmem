#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=play-eval
#SBATCH --output=logs/play_eval_%x_%j.out
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
# CKPT, USE_LEX, RETR via --export
NPROC=${NPROC:-60} python play_eval.py ${NG:-2000}
echo "=== PLAY EVAL DONE"
