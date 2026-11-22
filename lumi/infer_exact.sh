#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=infer-exact
#SBATCH --output=logs/infer_exact_%j.out
# Rack-inference references (round 6), CPU only:
#  1. exact inversion of a static HastyBot (go/lexmem_invert) on all 3000 held-out positions -> data/inferinv_test.jsonl
#  2. Macondo's rangefinder, two diagnostic settings, on the first 1000 (as data/inferref_test.jsonl, budget 200):
#     refine rounds drawing from the prior (PROPOSAL=prior) and a softer pinned temperature (TAU=0.3)
source lumi/common.sh
python eval/infer_macondo.py prep 3000 || exit 1   # the first 1000 lines are the ones the earlier reference used
M=/scratch/project_462001611/macondo
(cd $M && WORKERS=128 ./bin/lexmem_invert < $PROJ/data/inferpos_test.jsonl > $PROJ/data/inferinv_test.jsonl) || exit 1
REF=inferinv_test python eval/infer_macondo.py analyze || exit 1
head -1000 data/inferpos_test.jsonl > data/inferpos_test1000.jsonl
(cd $M && PROPOSAL=prior WORKERS=32 THREADS=4 ./bin/lexmem_infer < $PROJ/data/inferpos_test1000.jsonl > $PROJ/data/inferref_test_prior.jsonl) || exit 1
(cd $M && TAU=0.3 WORKERS=32 THREADS=4 ./bin/lexmem_infer < $PROJ/data/inferpos_test1000.jsonl > $PROJ/data/inferref_test_tau03.jsonl) || exit 1
for r in inferref_test inferref_test_prior inferref_test_tau03; do REF=$r python eval/infer_macondo.py analyze || exit 1; done
echo "=== INFER EXACT DONE"
