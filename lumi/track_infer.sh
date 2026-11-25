#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=track-infer
#SBATCH --output=logs/track_infer_%j.out
# Rack tracking over the whole move history (round 7; eval/track_infer.py, go/lexmem_track), CPU only.
# 1. the opponent's moves in every held-out position's game -> data/trackpos_test.jsonl
# 2. check: the first CHECK positions with FAST=0 (every rack on the full move list) against the default
# 3. exact tracking with the last 1 / 2 / 4 moves and the whole history (HISTS; 0 = whole, give it last)
#    -> data/trackinv_h<k>.jsonl with 256 posterior samples; HIST=1 is checked against go/lexmem_invert (data/inferinv_test.jsonl).
#    The HISTS runs share the cores (all at once, W workers each: a chain with exchange steps runs for tens of minutes on one
#    core), so their "secs" are wall-clock times under load.
# 4. the simulator test of lumi/sim_infer.sh with these sources as new arms (SIMH; arm track<k>_10 / trackall10, 10 s per
#    position), then every arm's choice (OLD arms from data/simrack_<arm>.jsonl) evaluated together under the true kept tiles
# Env: NPOS (default 3000) WORKERS HISTS (default "1 2 4 0") SIMH (default "2 0"; "" = no simulator part) CHECK (default 200;
#      0 = none) ITERS (evaluation iterations per play, default 6000) OLD PREFIX
source lumi/common.sh
M=/scratch/project_462001611/macondo
N=${NPOS:-3000}; W=${WORKERS:-$((SLURM_CPUS_PER_TASK - 8))}; HISTS=${HISTS-"1 2 4 0"}; SIMH=${SIMH-"2 0"}; C=${CHECK:-200}; export PREFIX=${PREFIX:-simrack}
OLD=${OLD-"uni10 uni10b uni3 uni30 true10 exact10 lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
python eval/track_infer.py prep $N || exit 1
track() {   # HIST, input file, output name, extra environment
  (cd $M && env $4 HIST=$1 SAMPLES=256 WORKERS=$W ./bin/lexmem_track < $PROJ/data/$2 2> $PROJ/logs/track_$3_${SLURM_JOB_ID}.err) > data/$3.jsonl || exit 1
  echo "[$3] $(tail -1 logs/track_$3_${SLURM_JOB_ID}.err)"
}
if [ "$C" -gt 0 ]; then
  head -$C data/trackpos_test.jsonl > data/trackpos_check.jsonl
  track 0 trackpos_check.jsonl trackchk_fast; track 0 trackpos_check.jsonl trackchk_slow FAST=0
  python eval/track_infer.py check trackchk_fast trackchk_slow || exit 1
fi
REFS=""; PIDS=""
for h in $HISTS; do track $h trackpos_test.jsonl trackinv_h$h & PIDS="$PIDS $!"; REFS="$REFS trackinv_h$h"; done
for p in $PIDS; do wait $p || exit 1; done
case " $HISTS " in *" 1 "*) python eval/track_infer.py check trackinv_h1 inferinv_test || exit 1;; esac
python eval/track_infer.py analyze inferinv_test $REFS || exit 1
NEW=""
for h in $SIMH; do
  k=$h; [ $h = 0 ] && k=all; a=track${k}_10; [ $h = 0 ] && a=trackall10
  python eval/track_infer.py okeep trackinv_h$h track$k || exit 1
  head -$N data/planinf_track$k.jsonl | (cd $M && SECS=10 WORKERS=$W ./bin/lexmem_simrack 2> $PROJ/logs/${PREFIX}_${a}_${SLURM_JOB_ID}.err) > data/${PREFIX}_$a.jsonl || exit 1
  echo "[$a] $(tail -1 logs/${PREFIX}_${a}_${SLURM_JOB_ID}.err)"; NEW="$NEW $a"
done
if [ -n "$NEW" ]; then
  [ -f data/${PREFIX}_eval_v1.jsonl ] || cp data/${PREFIX}_eval.jsonl data/${PREFIX}_eval_v1.jsonl   # lumi/sim_infer.sh's evaluation (OLD arms only)
  /usr/bin/python3 eval/sim_infer.py prep-eval $OLD $NEW || exit 1
  (cd $M && MODE=eval ITERS=${ITERS:-6000} WORKERS=$W ./bin/lexmem_simrack < $PROJ/data/${PREFIX}_evalin.jsonl 2> $PROJ/logs/${PREFIX}_eval_${SLURM_JOB_ID}.err) > data/${PREFIX}_eval.jsonl || exit 1
  echo "[eval] $(tail -1 logs/${PREFIX}_eval_${SLURM_JOB_ID}.err)"
  /usr/bin/python3 eval/sim_infer.py analyze $OLD $NEW || exit 1
  echo "--- paired against the last move alone (exact10)"
  /usr/bin/python3 eval/sim_infer.py analyze exact10 $NEW true10 || exit 1
fi
echo "=== TRACK INFER DONE"
