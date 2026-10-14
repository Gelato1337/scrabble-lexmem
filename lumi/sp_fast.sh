#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sp_fast_%x_%j.out
# Fast self-play on one GPU (GCD): server_infer.py on the GPU + Go engine (bin/lexmem_selfplay) on the node's cores.
# Env: CKPT NGAMES CONC BATCH PTEMP SEED OUT PARITY
source lumi/common.sh
export PYTHONPATH=. SOCK=/tmp/lexmem_$SLURM_JOB_ID.sock
python selfplay/server_infer.py > logs/sp_server_$SLURM_JOB_ID.log 2>&1 &
SRV=$!
for i in $(seq 120); do [ -S $SOCK ] && break; sleep 2; done
[ -S $SOCK ] || { echo "server did not start"; cat logs/sp_server_$SLURM_JOB_ID.log; exit 1; }
cd /scratch/project_462001611/macondo
LEXWORDS=/scratch/project_462001611/scrabble-lexmem/data/lexwords.txt GOMAXPROCS=${GOMAXPROCS:-7} ./bin/lexmem_selfplay
RC=$?
cd /scratch/project_462001611/scrabble-lexmem; kill $SRV; tail -3 logs/sp_server_$SLURM_JOB_ID.log
[ -n "$PARITY_CHECK" ] && python eval/parity_check.py $OUT 300
exit $RC
