#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=hasty-gen
#SBATCH --output=logs/hasty_gen_%j.out
NG=${NG:-100000}; TAG=${TAG:-hasty100k}
cd /scratch/project_462001611/macondo
mkdir -p experiments/$TAG
(echo "autoplay -botcode1 HASTY_BOT -botcode2 HASTY_BOT -numgames $NG -threads 60 -outputdir ./experiments/$TAG -lexicon NWL23"
 # wait until the games file has NG lines (+ header)
 while true; do sleep 30; g=$(ls experiments/$TAG/games-*.txt 2>/dev/null | head -1); [ -n "$g" ] && [ $(wc -l < $g) -ge $((NG+1)) ] && break; done
 echo "autoplay stop"; sleep 5; echo exit) | ./bin/shell 2>&1 | grep -v "level\|INFO\|DEBUG" | tail -5
wc -l experiments/$TAG/*.txt
echo "=== HASTY GEN DONE"
