#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=infer-diag
#SBATCH --output=logs/infer_diag_%j.out
# Macondo's rangefinder (go/lexmem_infer; round 6), two diagnostic settings, CPU only, on the first 1000 held-out positions
# (as data/inferref_test.jsonl, budget 200): refine rounds drawing from the prior (PROPOSAL=prior) and a softer pinned
# temperature (TAU=0.3); analysed with the default run and the nets.
# Env: TAGS (rack-inference nets with data/<TAG>_evbits.npy)
source lumi/common.sh
TAGS=${TAGS:-"lexinfer_fix_lex_all_s0 lexinfer_fix_lex_all_s1 lexinfer_fix_nolex_all_s0 lexinfer_fix_nolex_all_s1"}
python eval/infer_macondo.py prep 3000 || exit 1   # the first 1000 lines are the ones the earlier reference used
head -1000 data/inferpos_test.jsonl > data/inferpos_test1000.jsonl
M=/scratch/project_462001611/macondo
(cd $M && PROPOSAL=prior WORKERS=32 THREADS=4 ./bin/lexmem_infer < $PROJ/data/inferpos_test1000.jsonl > $PROJ/data/inferref_test_prior.jsonl) || exit 1
(cd $M && TAU=0.3 WORKERS=32 THREADS=4 ./bin/lexmem_infer < $PROJ/data/inferpos_test1000.jsonl > $PROJ/data/inferref_test_tau03.jsonl) || exit 1
for r in inferref_test inferref_test_prior inferref_test_tau03; do REF=$r python eval/infer_macondo.py analyze $TAGS || exit 1; done
echo "=== INFER DIAG DONE"
