#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=plan-infer
#SBATCH --output=logs/plan_infer_%j.out
# Planning with rack inference (round 6, eval/plan_infer.py), CPU only: the one-turn exact planner on the 3000 held-out
# rack-inference positions, the opponent's rack in each sample from each inference source; regret under the true kept tiles.
# Env: TAGS (rack-inference nets with data/<TAG>_evlogp.npy)
source lumi/common.sh
M=/scratch/project_462001611/macondo
TAGS=${TAGS:-"lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
# the exact inversion's posterior samples: data/inferinv_test.jsonl from lumi/infer_exact.sh (submit with --dependency on it)
TAGS="$TAGS" NPROC=126 python eval/plan_infer.py prep 3000 || exit 1
plan() { (cd $M && LEXWORDS=$PROJ/data/lexwords.txt K=16 S=256 WORKERS=126 SEED0=${2:-0} ./bin/lexmem_plan < $PROJ/data/planinf_$1.jsonl > $PROJ/data/planinf_out_$3.jsonl) || exit 1; echo "labelled $3: $(wc -l < data/planinf_out_$3.jsonl)"; }
for c in uniform base exact true $TAGS; do plan $c 0 $c; done
plan true 1000003 true_eval   # independent samples: the yardstick
python eval/plan_infer.py analyze uniform base exact true $TAGS || exit 1
echo "=== PLAN INFER DONE"
