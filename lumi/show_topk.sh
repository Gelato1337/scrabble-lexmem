#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=show-topk
#SBATCH --output=logs/show_topk_%j.out
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=. BOT_MEM=1GiB
python show_topk.py ${OUT:-docs/samples} || exit 1
echo "=== SHOW DONE"
