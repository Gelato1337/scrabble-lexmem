#!/bin/bash
# usage: bash lumi/round.sh <round> <variant: nolex|lex> <init ckpt> [dependency jobid]
# submits: self-play gen (CPU) -> train (GPU) -> eval with search + eval without (CPU)
R=$1; V=$2; INIT=$3; DEP=$4
cd /scratch/project_462001611/scrabble-lexmem
UL=0; RETR=""; [ "$V" = "lex" ] && UL=1 && RETR=data/lexmem_nwl.pt
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
G=$(sbatch --parsable $D --export=ALL,CKPT=$INIT,USE_LEX=$UL,TAG=sp${R}_${V},NG=${NG:-15000},SEED0=$((100000 + R * 100000)) --job-name=sp${R}-gen-${V} lumi/selfplay_gen.sh)
T=$(sbatch --parsable --dependency=afterok:$G --export=ALL,USE_LEX=$UL,TAG=sp${R}_${V},RETR=$RETR,INIT=$INIT,TRAIN=sp${R}_${V},TEST=hasty_test,STEPS=${STEPS:-8000},PLAY=0 --job-name=sp${R}-train-${V} lumi/imitate.sh)
E1=$(sbatch --parsable --dependency=afterok:$T --export=ALL,CKPT=data/policy_sp${R}_${V}.pt,USE_LEX=$UL,NG=2000,SEARCH=1 --job-name=sp${R}-eval-s-${V} lumi/play_eval.sh)
E0=$(sbatch --parsable --dependency=afterok:$T --export=ALL,CKPT=data/policy_sp${R}_${V}.pt,USE_LEX=$UL,NG=2000,SEARCH=0 --job-name=sp${R}-eval-${V} lumi/play_eval.sh)
echo "round $R $V: gen $G train $T eval-search $E1 eval $E0"
