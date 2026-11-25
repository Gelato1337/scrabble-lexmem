#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=224G
#SBATCH --time=04:00:00
#SBATCH --job-name=late-game
#SBATCH --output=logs/late_game_%j.out
# Exact solving earlier, late-game playouts (eval/late_game.py, go/lexmem_search START, go/lexmem_bot PEG_UNSEEN), CPU only.
# Positions with BAG tiles in the bag (our side on turn, both racks known) are played to the end against HastyBot by SIMMING_BOT
# as it is (run "sim": simulation now, its solvers from 8 unseen tiles) and with Macondo's pre-endgame solver from this position
# on (runs "peg<seconds>": PEG_UNSEEN = BAG + 7, one thread, that many seconds at the early stage). Every other move: 10 s.
# Env: BAG (default 2) NPOS (default 1000) PEGS (seconds of the solver runs, default "10 60") NBOT (bot processes, default 60)
#      PEG_THREADS (solver threads, default 1) with BOT_PROCS (GOMAXPROCS of a bot process) and BOT_MEM (its GOMEMLIMIT)
source lumi/common.sh; set -o pipefail
M=/scratch/project_462001611/macondo
BAG=${BAG:-2}; N=${NPOS:-1000}; NB=${NBOT:-60}
python eval/late_game.py prep $BAG $N || exit 1
run() {   # tag, extra environment
  local tag=late${BAG}_$1; shift; rm -f logs/bot_${tag}_${SLURM_JOB_ID}.err
  (cd $M && env "$@" START=$PROJ/data/late_bag$BAG.jsonl OURS=SIMMING_BOT OURS_SECS=10 OPP=HASTY_BOT NBOT=$NB CONC=$NB BOT_ERR=$PROJ/logs/bot_${tag}_${SLURM_JOB_ID}.err \
     LEXWORDS=$PROJ/data/lexwords.txt MACONDO=$M OUT=$PROJ/data/search_${tag}.jsonl ./bin/lexmem_search) 2> logs/search_${tag}_${SLURM_JOB_ID}.err | sed "s/^/[$tag] /" || return 1
  echo "[$tag] solver moves $(grep -c '^PEG ' logs/bot_${tag}_${SLURM_JOB_ID}.err), mean seconds $(awk '$1=="PEG"{s+=$3;n++}END{if(n)printf "%.1f",s/n}' logs/bot_${tag}_${SLURM_JOB_ID}.err), fallbacks to the static play $(grep -c FALLBACK logs/bot_${tag}_${SLURM_JOB_ID}.err)"
}
TAGS="late${BAG}_sim"
run sim || exit 1
for s in ${PEGS:-"10 60"}; do run peg$s PEG_UNSEEN=$((BAG + 7)) PEG_SECS=$s || exit 1; TAGS="$TAGS late${BAG}_peg$s"; done
/usr/bin/python3 eval/late_game.py analyze $TAGS || exit 1
echo "=== LATE GAME DONE"
