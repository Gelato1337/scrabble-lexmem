#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=2G
#SBATCH --time=24:00:00
#SBATCH --job-name=xt-crawl
#SBATCH --output=logs/xt_crawl_%j.out
# cross-tables GCG download (1 req/s, resumable: existing files are skipped). Index is already complete.
# Stdlib only: runs on the node's system python, not the pytorch container (its image load can hang for minutes).
cd /scratch/project_462001611/scrabble-lexmem
export PYTHONUNBUFFERED=1
/usr/bin/python3 prep/xt_crawl.py gcg ${LEXICONS:-NWL,TWL}
echo "=== CRAWL DONE $(ls data/xt/gcg | wc -l) files"
