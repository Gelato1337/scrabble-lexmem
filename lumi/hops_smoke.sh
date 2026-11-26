#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu-paths-smoke
#SBATCH --output=logs/gpu_paths_smoke_%j.out
# CPU-only smoke test of the two GPU experiments (no GPU hours): the several-lookups planning net (train_lexvalue.py HOPS) and the
# bigger policy net (imitate.py D / LAYERS, eval/policy_agree.py, eval/play_eval.py with the width read from the checkpoint).
# REG=1 (default): the round-7 checkpoint lexplanfix_h3ft_lex_all_s0 re-evaluated with the changed code (HOPS=1 path) on the
# 3000 held-out positions, to compare with its stored last evaluation (fp32 on CPU against bf16 on the GPU: equal to rounding).
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=. OMP_NUM_THREADS=8
set -o pipefail
P="HORIZON=3 DATA=data/plan_h3.pkl QINIT=fut_train NO_XCHK=1 HOLDOUT_MOD=10"
SMK="$P NWORK=8 NEV_POS=40 EVAL_EVERY=20 MAX_TRAIN=200 STEPS=20 B=16"
F='trainable|query encoder|^\[step|^\[init|train positions'
env $SMK ARM=lex HOPS=2 TAG=hop_smoke_a python train/train_lexplan.py | grep -E "$F" || exit 1
env $SMK ARM=lex HOPS=2 HOP_LEX=0 TAG=hop_smoke_b python train/train_lexplan.py | grep -E "$F" || exit 1
env $SMK ARM=nolex HOPS=2 TAG=hop_smoke_c python train/train_lexplan.py | grep -E "$F" || exit 1
env $SMK ARM=lex HOPS=3 HOP_LAYERS=2 TAG=hop_smoke_d python train/train_lexplan.py | grep -E "$F" || exit 1
python eval/plan_pairs.py hop_smoke_b hop_smoke_a hop_smoke_c,hop_smoke_d || exit 1
echo "--- bigger net"
TRAIN=simsoft_0 TEST=simsoft_31 LEXICON=data/NWL23.txt USE_LEX=0 EI=1 TAU=1.0 MARGIN=0 BETA=0 STEPS=6 B=8 NEV=16 EVAL_EVERY=6 PLAY=0 NWORK=8 D=512 LAYERS=6 \
  TAG=bignet_smoke python train/imitate.py | grep -E "net D=|params|\[final|^step" || exit 1
TEST=simsoft_31 N=400 NWORK=24 python eval/policy_agree.py data/policy_simsoft_scratch_nolex.pt data/policy_bignet_smoke.pt || exit 1
CKPT=data/policy_bignet_smoke.pt USE_LEX=0 NPROC=8 TAGE=_smoke python eval/play_eval.py 8 777000 | tail -1 || exit 1
if [ "${REG:-1}" = 1 ]; then
  echo "--- stored last evaluation of lexplanfix_h3ft_lex_all_s0"; grep -h "^\[step 40000\]" logs/lexplanfix_h3ft_lex_all_s0_*.out
  echo "--- the same checkpoint, changed code, CPU"
  OMP_NUM_THREADS=24 env $P ARM=lex NWORK=28 NEV_POS=3000 EVAL_CKPTS=data/lexplanfix_h3ft_lex_all_s0.pt python train/train_lexplan.py | grep -E "^\[|eval candidates" || exit 1
fi
echo "=== SMOKE DONE"
