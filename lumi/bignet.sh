#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=bignet
#SBATCH --output=logs/bignet_%x_%j.out
# Bigger policy net: the recipe of simsoft_scratch_nolex (README: soft teacher targets, 715k positions, 30000 steps of 48, from
# scratch, no lexicon branch) with width D and LAYERS layers instead of 256 and 3. Then the reference net and the new one against
# the teacher on the same NAGREE held-out positions (eval/policy_agree.py). No games here (CPU: eval/play_eval.py).
# Env: D (512) LAYERS (6) STEPS (30000) TAG REF NAGREE (5000).
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
D=${D:-512}; LAYERS=${LAYERS:-6}; TAG=${TAG:-simsoft_scratch_nolex_d${D}l${LAYERS}}; REF=${REF:-data/policy_simsoft_scratch_nolex.pt}
TRAIN=$(seq -s, -f "simsoft_%g" 0 30) TEST=simsoft_31 LEXICON=data/NWL23.txt USE_LEX=0 EI=1 TAU=1.0 MARGIN=0 BETA=0 STEPS=${STEPS:-30000} B=48 \
  NEV=600 EVAL_EVERY=2000 PLAY=0 D=$D LAYERS=$LAYERS TAG=$TAG python train/imitate.py || exit 1
TEST=simsoft_31 N=${NAGREE:-5000} OUT=data/agree_$TAG.pkl python eval/policy_agree.py $REF data/policy_$TAG.pt || exit 1
echo "=== BIGNET DONE"
