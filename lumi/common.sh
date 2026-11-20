# sourced by every job
export PROJ=/scratch/project_462001611/scrabble-lexmem
cd $PROJ
export PYTHONPATH=$PROJ   # scripts live in prep/ train/ selfplay/ eval/ feasibility/ and import lexmem from the repo root
module use /appl/local/csc/modulefiles
module load pytorch/2.7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p logs data
# own MIOpen db per job: a stale db of another user in a node's shared /tmp crashes the GPU process (nid005093, nid005114)
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID:-$$} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID:-$$}; mkdir -p $MIOPEN_USER_DB_PATH
