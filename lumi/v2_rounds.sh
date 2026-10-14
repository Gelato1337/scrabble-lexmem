#!/bin/bash
# Submit v2 rounds R0..R1 for one arm: gen (NODES x NGT games, from the current best net) -> train -> eval + gate.
# Replay buffer = this round + the previous one. First call: initialise data/v2_best_${ARM}.txt / .score by hand.
# Usage: ARM=w NODES=4 NGT=12500 ARM_SEED=30000000 bash lumi/v2_rounds.sh R0 R1 [DEP]
R0=$1; R1=$2; DEP=$3
cd /scratch/project_462001611/scrabble-lexmem
for R in $(seq $R0 $R1); do
  D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
  GEN=$(sbatch --parsable $D --array=0-$((NODES-1)) -J v2-$ARM-g$R-gen --export=ALL,ARM=$ARM,R=$R,NGT=$NGT,SEEDBASE=$((ARM_SEED + R * 1000000)) lumi/v2_gen.sh)
  TR=$(seq -s, -f "v2_${ARM}_g${R}_%g" 0 $((NODES-1)))
  [ "$R" -gt 1 ] && TR=$TR,$(seq -s, -f "v2_${ARM}_g$((R-1))_%g" 0 $((NODES-1)))
  L1=0.7; [ "$R" = 1 ] && L1=1
  TRN=$(TRAIN=$TR sbatch --parsable --dependency=afterok:$GEN -J v2-$ARM-g$R-train --export=ALL,ARM=$ARM,R=$R,LAM1=$L1 lumi/v2_train.sh)
  EV=$(sbatch --parsable --dependency=afterok:$TRN -J v2-$ARM-g$R-eval --export=ALL,ARM=$ARM,R=$R lumi/v2_eval.sh)
  echo "round $R: gen=$GEN train=$TRN eval=$EV"; DEP=$EV
done
