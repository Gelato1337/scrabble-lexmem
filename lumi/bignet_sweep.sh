#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 --mem=160G
#SBATCH --time=08:00:00
#SBATCH --job-name=bignet-sweep
#SBATCH --output=logs/bignet_sweep_%j.out
# Several policy nets at once on one stream of batches (train/imitate_sweep.py), spread over 4 GPUs: 4 GPUs with 32 cores bill
# what 1 GPU with 32 cores bills (2 GPU-h per hour; the 32 data workers need the cores). Recipe of simsoft_scratch_nolex, per net
# its own width, depth, learning rate and warm-up. Then the reference net and every new one against the teacher on the same NAGREE
# held-out positions (eval/policy_agree.py). Games: lumi/bignet_games.sh (CPU).
# Env: TRAIN (game sets, default simsoft_0-30), SWEEP ("tag:D:LAYERS:LR:WARMUP ...", default below; net i on GPU i modulo 4) STEPS REF NAGREE MAXH (hours, default 6).
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
P=simsoft_scratch_nolex
SWEEP=${SWEEP:-"${P}_d384l4_lr2e4:384:4:2e-4:1000 ${P}_d512l6_lr1e4:512:6:1e-4:1000 ${P}_d512l6_lr5e5:512:6:5e-5:1000 ${P}_d512l6_lr2e4:512:6:2e-4:2000 ${P}_d256l3_lr1e4:256:3:1e-4:1000"}
REF=${REF:-data/policy_$P.pt}
TRAIN=${TRAIN:-$(seq -s, -f "simsoft_%g" 0 30)}   # 2.0M positions: add simsoftb_0-35 and simsoftc_0-21 (needs --mem=250G)
SWEEP="$SWEEP" TRAIN=$TRAIN TEST=simsoft_31 LEXICON=data/NWL23.txt TAU=1.0 MARGIN=0 STEPS=${STEPS:-30000} B=48 \
  NEV=600 EVAL_EVERY=2000 NGPU=4 MAXH=${MAXH:-6} python train/imitate_sweep.py || exit 1
TEST=simsoft_31 N=${NAGREE:-5000} OUT=data/agree_sweep_$SLURM_JOB_ID.pkl timeout 45m python eval/policy_agree.py $REF $(for c in $SWEEP; do echo data/policy_${c%%:*}.pt; done) || exit 1
echo "=== BIGNET SWEEP DONE"
