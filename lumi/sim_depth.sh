#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=sim-depth
#SBATCH --output=logs/sim_depth_%j.out
# Deeper simulation, position level, CPU only (go/lexmem_simrack PLIES, eval/sim_infer.py): does SIMMING_BOT choose better when
# its rollouts look further ahead than 2 plies, at the same time limit and at three times it?
# New arms on the positions of lumi/sim_infer.sh (uniform opponent rack; arm p<plies>_<seconds>, pend = to the end of the
# game): the 2-ply arms are uni10 and uni30 of that job. Two evaluations of every arm's choice, both with the true kept tiles:
#   EVAL=      the 2-ply simulation of lumi/sim_infer.sh (6000 iterations per play; its leaf is the one the 2-ply arms use)
#   EVAL=_end  rollouts to the end of the game (PLIES=99, ITERS_END iterations per play): the value is the game's result when
#              both sides go on as static players, no leaf
# Env: NPOS (default 3000) WORKERS ARMS_NEW ("plies:seconds ...") OLD (arms already in data/simrack_<arm>.jsonl) ITERS ITERS_END
PROJ=/scratch/project_462001611/scrabble-lexmem; M=/scratch/project_462001611/macondo; cd $PROJ || exit 1
W=${WORKERS:-$((SLURM_CPUS_PER_TASK - 8))}; N=${NPOS:-3000}; export PREFIX=${PREFIX:-simrack}
OLD=${OLD-"uni10 uni10b uni3 uni30 true10 exact10 trackall10 lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
NEW=""
for a in ${ARMS_NEW-"4:10 6:10 4:30 6:30 99:30"}; do
  p=${a%%:*}; s=${a##*:}; n=p${p}_$s; [ $p = 99 ] && n=pend_$s
  head -$N data/planinf_uniform.jsonl | (cd $M && PLIES=$p SECS=$s WORKERS=$W ./bin/lexmem_simrack 2> $PROJ/logs/${PREFIX}_${n}_${SLURM_JOB_ID}.err) > data/${PREFIX}_$n.jsonl || exit 1
  echo "[$n] $(tail -1 logs/${PREFIX}_${n}_${SLURM_JOB_ID}.err)"; NEW="$NEW $n"
done
evaluate() {   # EVAL suffix, extra environment of the evaluation, iterations
  export EVAL=$1
  /usr/bin/python3 eval/sim_infer.py prep-eval $OLD $NEW || exit 1
  (cd $M && env $2 MODE=eval ITERS=$3 WORKERS=$W ./bin/lexmem_simrack < $PROJ/data/${PREFIX}_evalin$1.jsonl 2> $PROJ/logs/${PREFIX}_eval$1_${SLURM_JOB_ID}.err) > data/${PREFIX}_eval$1.jsonl || exit 1
  echo "[eval$1] $(tail -1 logs/${PREFIX}_eval$1_${SLURM_JOB_ID}.err)"
  /usr/bin/python3 eval/sim_infer.py analyze $OLD $NEW || exit 1
}
[ -f data/${PREFIX}_eval.jsonl ] && [ ! -f data/${PREFIX}_eval_v2.jsonl ] && cp data/${PREFIX}_eval.jsonl data/${PREFIX}_eval_v2.jsonl   # lumi/track_infer.sh's evaluation
echo "--- evaluation: rollouts to the end of the game"; evaluate _end PLIES=99 ${ITERS_END:-6000}
echo "--- evaluation: 2-ply simulation"; evaluate "" "" ${ITERS:-6000}
echo "=== SIM DEPTH DONE"
