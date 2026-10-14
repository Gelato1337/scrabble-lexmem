#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=label-bot
#SBATCH --output=logs/label_bot_%x_%A_%a.out
# CANDS=K stores the sim's top-K candidates (soft targets). G_OFF shifts the game range.
# Relabel HastyBot games with a stronger Macondo bot (default SIMMING_BOT, 10 s/move). Array index i -> games [i*GPT, (i+1)*GPT).
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=. BOT_MEM=${BOT_MEM:-1GiB}
I=${SLURM_ARRAY_TASK_ID:-0}; GPT=${GPT:-1000}; G_OFF=${G_OFF:-0}
TEACHER=${TEACHER:-SIMMING_BOT} SECS=${SECS:-10} NPROC=${NPROC:-64} CORES_PER_WORKER=2 SRC=${SRC:-hasty_train} \
  python label_bot.py $((G_OFF+I*GPT)) $((G_OFF+(I+1)*GPT)) ${TAG:-simbot}_$I || exit 1
echo "=== LABEL DONE"
