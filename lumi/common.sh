# sourced by every job
export PROJ=/scratch/project_462001611/scrabble-lexmem
cd $PROJ
module use /appl/local/csc/modulefiles
module load pytorch/2.7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p logs data
