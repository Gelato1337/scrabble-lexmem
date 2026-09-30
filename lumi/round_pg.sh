#!/bin/bash
# policy-gradient round: usage bash lumi/round_pg.sh <round> <nolex|lex> <init ckpt> [dependency]
R=$1; V=$2; INIT=$3; DEP=$4
cd /scratch/project_462001611/scrabble-lexmem
UL=0; RETR=""; [ "$V" = "lex" ] && UL=1 && RETR=data/lexmem_nwl.pt
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
G=$(sbatch --parsable $D --export=ALL,CKPT=$INIT,USE_LEX=$UL,TAG=pg${R}_${V},NG=${NG:-15000},SEED0=$((200000 + R * 100000)),SEARCH=0,TEMP=${TEMP:-0.7} --job-name=pg${R}-gen-${V} lumi/selfplay_gen.sh)
T=$(sbatch --parsable --dependency=afterok:$G --export=ALL,USE_LEX=$UL,TAG=pg${R}_${V},RETR=$RETR,INIT=$INIT,TRAIN=pg${R}_${V},TEST=hasty_test,STEPS=${STEPS:-3000},PLAY=0,PG=1,LR=${LR:-5e-5},TEMP=${TEMP:-0.7},KPLY=4 --job-name=pg${R}-train-${V} lumi/imitate.sh)
E0=$(sbatch --parsable --dependency=afterok:$T --export=ALL,CKPT=data/policy_pg${R}_${V}.pt,USE_LEX=$UL,NG=2000,SEARCH=0 --job-name=pg${R}-eval-${V} lumi/play_eval.sh)
echo "pg round $R $V: gen $G train $T eval $E0"
