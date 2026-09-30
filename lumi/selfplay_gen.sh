#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=sp-gen
#SBATCH --output=logs/spgen_%x_%j.out
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
# CKPT, USE_LEX, TAG, NG, SEED0 via --export
NPROC=${NPROC:-60} python selfplay.py ${NG:-15000} ${SEED0:-100000}
echo "=== SP GEN DONE"
