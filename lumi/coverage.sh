#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=100G
#SBATCH --time=06:00:00
#SBATCH --job-name=lex-coverage
#SBATCH --output=logs/coverage_%j.out
source lumi/common.sh
TEST=bigtest CKPT=data/lexmem_noslot.pt SLOT=0 OUT=data/coverage_noslot.pkl python eval/eval_coverage.py
echo "=== COVERAGE DONE"
