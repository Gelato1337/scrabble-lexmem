#!/bin/bash
# expert-iteration round: usage bash lumi/round_ei.sh <round> <nolex|lex> <init ckpt> [dependency]
R=$1; V=$2; INIT=$3; DEP=$4
cd /scratch/project_462001611/scrabble-lexmem
UL=0; RETR=""; [ "$V" = "lex" ] && UL=1 && RETR=data/lexmem_nwl.pt
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
G=$(sbatch --parsable $D --time=08:00:00 --export=ALL,NPROC=48,CKPT=$INIT,USE_LEX=$UL,TAG=ei${R}_${V},NG=${NG:-8000},SEED0=$((300000 + R * 100000)),SIM=1,CAND=${CAND:-both},ROLLOUT=${ROLLOUT:-hasty},K=${K:-4},S=${S:-24},S2=${S2:-96},VCKPT=${VCKPT:-},VLEX=${VLEX:-0} --job-name=ei${R}-gen-${V} lumi/selfplay_gen.sh)
T=$(sbatch --parsable --dependency=afterok:$G --export=ALL,USE_LEX=$UL,TAG=ei${R}_${V},RETR=$RETR,INIT=$INIT,TRAIN=ei${R}_${V},TEST=hasty_test,STEPS=${STEPS:-4000},PLAY=0,EI=1,LR=${LR:-1e-4},KPLY=4,TAU=${TAU:-2},MARGIN=${MARGIN:-3},BETA=${BETA:-0.5} --job-name=ei${R}-train-${V} lumi/imitate.sh)
E0=$(sbatch --parsable --dependency=afterok:$T --export=ALL,CKPT=data/policy_ei${R}_${V}.pt,USE_LEX=$UL,NG=2000,SEARCH=0 --job-name=ei${R}-eval-${V} lumi/play_eval.sh)
E1=$(sbatch --parsable --dependency=afterok:$T --time=10:00:00 --export=ALL,NPROC=48,CKPT=data/policy_ei${R}_${V}.pt,USE_LEX=$UL,NG=2000,SIM=1,CAND=net,ROLLOUT=hasty,K=4,S=24,S2=96,TAGE=_netcand --job-name=ei${R}-evalsim-${V} lumi/play_eval.sh)
echo "ei round $R $V: gen $G train $T eval $E0 evalsim $E1"
