#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/plan_label_%A_%a.out
# Planning-benchmark labels (go/lexmem_plan): array task i labels lines [i*PER, (i+1)*PER) of $IN. Env: IN PER K S TAG
source lumi/common.sh
I=${SLURM_ARRAY_TASK_ID:-0}
sed -n "$((I*PER+1)),$(((I+1)*PER))p" $IN > data/${TAG}_in_$I.jsonl
(cd /scratch/project_462001611/macondo && LEXWORDS=$PROJ/data/lexwords.txt K=${K:-16} S=${S:-256} WORKERS=126 ./bin/lexmem_plan < $PROJ/data/${TAG}_in_$I.jsonl > $PROJ/data/${TAG}_out_$I.jsonl) || exit 1
echo "=== PLAN LABEL DONE $(wc -l < data/${TAG}_out_$I.jsonl) positions"
