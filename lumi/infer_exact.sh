#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=infer-exact
#SBATCH --output=logs/infer_exact_%j.out
# Exact rack inference against a static HastyBot (go/lexmem_invert; round 6), CPU only: all 3000 held-out positions
# -> data/inferinv_test.jsonl, with 256 posterior samples per position (read by lumi/plan_infer.sh); analysed next to the nets.
# Env: TAGS (rack-inference nets with data/<TAG>_evbits.npy)
source lumi/common.sh
TAGS=${TAGS:-"lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
python eval/infer_macondo.py prep 3000 || exit 1
M=/scratch/project_462001611/macondo
(cd $M && SAMPLES=256 WORKERS=128 ./bin/lexmem_invert < $PROJ/data/inferpos_test.jsonl > $PROJ/data/inferinv_test.jsonl) || exit 1
REF=inferinv_test python eval/infer_macondo.py analyze $TAGS || exit 1
echo "=== INFER EXACT DONE"
