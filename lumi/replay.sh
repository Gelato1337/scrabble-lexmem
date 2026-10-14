#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=lex-replay
#SBATCH --output=logs/replay_%j.out
source lumi/common.sh
PYTHONPATH=. NPROC=60 python prep/replay_all.py /scratch/project_462001611/macondo/experiments/hasty100k hasty 2000
ls -la data/games_hasty_*.pkl
echo "=== REPLAY DONE"
