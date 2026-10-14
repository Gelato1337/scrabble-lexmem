#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sp_branch_gen_%A_%a.out
# Go self-play with shared-draw branch pairs, one GCD per array task; then convert to a branch-only pickle.
# Env: CKPT NGT (games per task) TAG P_BRANCH H BK
source lumi/common.sh
I=${SLURM_ARRAY_TASK_ID:-0}
export PYTHONPATH=. SOCK=/tmp/lexmem_${SLURM_JOB_ID}_$I.sock
python selfplay/server_infer.py > logs/sp_server_${SLURM_JOB_ID}_$I.log 2>&1 &
SRV=$!
for i in $(seq 120); do [ -S $SOCK ] && break; sleep 2; done
[ -S $SOCK ] || { echo "server did not start"; cat logs/sp_server_${SLURM_JOB_ID}_$I.log; exit 1; }
OUT=/scratch/project_462001611/scrabble-lexmem/data/${TAG}_$I.jsonl
(cd /scratch/project_462001611/macondo && LEXWORDS=/scratch/project_462001611/scrabble-lexmem/data/lexwords.txt GOMAXPROCS=7 \
  NGAMES=$NGT CONC=1024 BATCH=2048 PTEMP=1.0 SEED=$((7000000 + I * 1000000)) OUT=$OUT ./bin/lexmem_selfplay) || { kill $SRV; exit 1; }
kill $SRV
BRANCH_ONLY=1 python prep/go2pkl.py $OUT ${TAG}_$I || exit 1
echo "=== BRANCH GEN DONE"
