#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/search_eval_%j.out
# Whole games in Go (cmd/lexmem_search) vs a Macondo bot, net served on one GCD by server_infer.py.
# Env: CKPT (net), TAG (output data/search_<TAG>.jsonl), VHEAD (val for LEAF=net), plus lexmem_search's own env
# (NGAMES SEED CONC OURS OPP HANDOFF EG_SECS NBOT K S D LEAF PM REC EXCH POSITIONS ...). Several runs: RUNS="TAG1|env1;TAG2|env2".
# PRE / POST: shell commands run before the server starts / after the runs (e.g. eval/search_positions.py prep / analyze).
source lumi/common.sh; set -o pipefail
export PYTHONPATH=. SOCK=/tmp/lexmem_${SLURM_JOB_ID}.sock VHEAD=${VHEAD:-val}
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}; mkdir -p $MIOPEN_USER_DB_PATH   # a stale MIOpen db of another user in /tmp crashed the server
[ -n "$PRE" ] && { eval "$PRE" || exit 1; }
python selfplay/server_infer.py > logs/search_server_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!
for i in $(seq 300); do [ -S $SOCK ] && break; sleep 2; done   # up to 10 min: model load can be slow on a busy node
[ -S $SOCK ] || { echo "server did not start"; cat logs/search_server_${SLURM_JOB_ID}.log; exit 1; }
run() {   # $1 = tag, rest = env assignments
  local tag=$1; shift
  (cd /scratch/project_462001611/macondo && env "$@" LEXWORDS=/scratch/project_462001611/scrabble-lexmem/data/lexwords.txt \
     MACONDO=/scratch/project_462001611/macondo OUT=/scratch/project_462001611/scrabble-lexmem/data/search_${tag}.jsonl \
     ./bin/lexmem_search) 2> logs/search_${tag}_${SLURM_JOB_ID}.err | sed "s/^/[$tag] /" || return 1
  echo "[$tag] NOMATCH $(grep -c NOMATCH logs/search_${tag}_${SLURM_JOB_ID}.err)"   # grep -c exits 1 on zero matches: not a failure
}
if [ -n "$RUNS" ]; then
  IFS=';' read -ra RS <<< "$RUNS"
  for r in "${RS[@]}"; do IFS='|' read -r tag envs <<< "$r"; run $tag $envs || { kill $SRV; exit 1; }; done
else
  run ${TAG:-test} || { kill $SRV; exit 1; }
fi
grep "positions/s" logs/search_server_${SLURM_JOB_ID}.log | tail -2
[ -n "$POST" ] && eval "$POST"
kill $SRV
echo "=== SEARCH EVAL DONE"
