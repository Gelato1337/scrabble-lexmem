#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2_eval_%x_%j.out
# Eval round R vs HastyBot (4000 games, paired seeds 777000, policy argmax, exchanges on) and gate: the new net becomes the
# arm's best only if its win rate >= the best so far on the same seeds. Env: ARM R
set -o pipefail
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
C=data/policy_v2_${ARM}_g$R.pt
CKPT=$C ACT=policy EXCH=1 OPP=hasty TAGE=_v2 NPROC=60 python eval/play_eval.py 4000 777000 | tee logs/v2_eval_${ARM}_g$R.txt || exit 1
NEW=$(grep "^FINAL" logs/v2_eval_${ARM}_g$R.txt | sed -E "s/.*win rate ([0-9.]+).*/\1/"); BEST=$(cat data/v2_best_${ARM}.score)
if python -c "import sys; sys.exit(0 if float('$NEW') >= float('$BEST') else 1)"; then
  echo $C > data/v2_best_${ARM}.txt; echo $NEW > data/v2_best_${ARM}.score; echo "GATE: accepted $C ($NEW >= $BEST)"
else echo "GATE: rejected $C ($NEW < $BEST); best stays $(cat data/v2_best_${ARM}.txt)"; fi
echo "=== V2 EVAL DONE"
