#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=future
#SBATCH --output=logs/future_%x_%j.out
source lumi/common.sh
export PYTHONPATH=.
# R, TAG via --export
LABELS=${LABELS:-data/simlabels_fut_hasty.pkl,data/simlabels_fut_ei1.pkl} STEPS=${STEPS:-12000} B=48 EVAL_EVERY=2000 NEV=2000 RETR=data/lexmem_nwl.pt python train_future.py
echo "=== FUTURE DONE"
