#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=160G
#SBATCH --time=12:00:00
#SBATCH --job-name=label-sim
#SBATCH --output=logs/label_%x_%j.out
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
NPROC=48 FUT=${FUT:-0} TAG=${TAG:-hasty300k} K=${K:-8} S=${S:-64} GAMES=${GAMES:-data/games_hasty_train.pkl} python prep/label_sim.py ${NPOS:-300000} ${SEED0:-0}
echo "=== LABEL DONE"
