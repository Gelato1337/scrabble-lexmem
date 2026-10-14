#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=lex-imitate
#SBATCH --output=logs/imitate_%x_%j.out
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
# USE_LEX, TAG, RETR via --export
LEXICON=data/NWL23.txt TRAIN=${TRAIN:-hasty_train} TEST=${TEST:-hasty_test} STEPS=${STEPS:-30000} B=48 NEV=600 EVAL_EVERY=2000 PLAY=${PLAY:-300} python imitate.py || exit 1
echo "=== IMITATE DONE"
