#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=sim-infer
#SBATCH --output=logs/sim_infer_%j.out
# Rack modelling inside Macondo's simulator, position level, CPU only (eval/sim_infer.py, go/lexmem_simrack).
# Needs data/planinf_<cond>.jsonl (eval/plan_infer.py prep: lumi/plan_infer.sh) for uniform, true, exact and the net TAGS.
# Arms: SIMMING_BOT's simulation with a uniform opponent rack at 10 s (twice: the second run is the noise floor), 3 s and 30 s;
# 10 s with the kept tiles from the exact HastyBot inversion, from each rack-inference net, and the true kept tiles.
# Env: NPOS (default 3000) WORKERS (default cpus - 8) ITERS (evaluation iterations per play, default 6000) TAGS PREFIX.
PROJ=/scratch/project_462001611/scrabble-lexmem; M=/scratch/project_462001611/macondo; cd $PROJ || exit 1
W=${WORKERS:-$((SLURM_CPUS_PER_TASK - 8))}; N=${NPOS:-3000}; export PREFIX=${PREFIX:-simrack}
TAGS=${TAGS:-"lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
run() {   # arm, condition (data/planinf_<condition>.jsonl), seconds per position
  head -$N data/planinf_$2.jsonl | (cd $M && SECS=$3 WORKERS=$W ./bin/lexmem_simrack 2> $PROJ/logs/${PREFIX}_$1_${SLURM_JOB_ID}.err) > data/${PREFIX}_$1.jsonl || exit 1
  echo "[$1] $(tail -1 logs/${PREFIX}_$1_${SLURM_JOB_ID}.err)"
}
ARMS="uni10 uni10b uni3 uni30 true10 exact10"
run uni10 uniform 10; run uni10b uniform 10; run uni3 uniform 3; run uni30 uniform 30
run true10 true 10; run exact10 exact 10
for t in $TAGS; do run $t $t 10; ARMS="$ARMS $t"; done
/usr/bin/python3 eval/sim_infer.py prep-eval $ARMS || exit 1
(cd $M && MODE=eval ITERS=${ITERS:-6000} WORKERS=$W ./bin/lexmem_simrack < $PROJ/data/${PREFIX}_evalin.jsonl 2> $PROJ/logs/${PREFIX}_eval_${SLURM_JOB_ID}.err) > data/${PREFIX}_eval.jsonl || exit 1
echo "[eval] $(tail -1 logs/${PREFIX}_eval_${SLURM_JOB_ID}.err)"
/usr/bin/python3 eval/sim_infer.py analyze $ARMS || exit 1
echo "=== SIM INFER DONE"
