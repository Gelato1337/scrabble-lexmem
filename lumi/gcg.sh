#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=gcg-replay
#SBATCH --output=logs/gcg_%j.out
source lumi/common.sh
PYTHONPATH=. NPROC=60 LEXICON=data/NWL23.txt python prep/gcg_all.py data/xt/gcg human 1000
ls -la data/games_human_*.pkl
echo "=== GCG DONE"
