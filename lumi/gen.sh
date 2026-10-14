#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=lex-gen
#SBATCH --output=logs/gen_%j.out
source lumi/common.sh
NPROC=62 python prep/gen_games.py 4000 big 1000
NPROC=62 python prep/gen_games.py 300 bigtest 900000
echo "=== GEN DONE"
