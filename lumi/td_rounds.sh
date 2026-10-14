#!/bin/bash
# Submit Track A rounds R0..R1 for one arm as a dependency chain: gen (NODES x NGT games) -> TD train -> eval vs HastyBot.
# Usage: ARM=w ACT=qtop ACT1=policy LAM1=1 NODES=4 NGT=12500 bash lumi/td_rounds.sh R0 R1 INIT [DEP]
#   ACT1/LAM1 apply to round 1 only (warm start acts with the policy head; the untrained V cannot bootstrap yet -> lambda 1).
#   Replay buffer = this round + the previous one. Eval: 2000 games, paired fresh seeds 777000, acting with EVACT.
R0=$1; R1=$2; INIT=$3; DEP=$4
cd /scratch/project_462001611/scrabble-lexmem
for R in $(seq $R0 $R1); do
  if [ "$R" = 1 ]; then A=${ACT1:-$ACT}; L=${LAM1:-1}; else A=$ACT; L=${LAM:-0.7}; fi
  D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
  GEN=$(sbatch --parsable $D --array=0-$((NODES-1)) -J td-$ARM-g$R-gen --export=ALL,ARM=$ARM,R=$R,CKPT=$INIT,ACT=$A,NGT=$NGT,SEEDBASE=$((ARM_SEED + R * 1000000)) lumi/td_gen.sh)
  TR=$(seq -s, -f "td_${ARM}_g${R}_%g" 0 $((NODES-1)))
  [ "$R" -gt 1 ] && TR=$TR,$(seq -s, -f "td_${ARM}_g$((R-1))_%g" 0 $((NODES-1)))
  TRN=$(TRAIN=$TR sbatch --parsable --dependency=afterok:$GEN -J td-$ARM-g$R-train --export=ALL,INIT=$INIT,TAG=td_${ARM}_g$R,LAM=$L,STEPS=${STEPS:-10000},LR=${LR:-1e-4} lumi/td_train.sh)
  EV=$(sbatch --parsable --dependency=afterok:$TRN -A project_462001611 -p small --cpus-per-task=64 --mem=120G -t 03:00:00 -J td-$ARM-g$R-eval -o logs/td_eval_${ARM}_g${R}_%j.out \
       --wrap="source lumi/common.sh; export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=. CKPT=data/policy_td_${ARM}_g$R.pt ACT=$EVACT EXCH=1 OPP=hasty TAGE=_td; NPROC=60 python eval/play_eval.py 2000 777000 || exit 1")
  echo "round $R: gen=$GEN train=$TRN eval=$EV (act=$A lambda=$L)"
  INIT=data/policy_td_${ARM}_g$R.pt; DEP=$TRN
done
