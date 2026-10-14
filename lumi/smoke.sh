#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=dev-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60G
#SBATCH --time=00:25:00
#SBATCH --job-name=lex-smoke
#SBATCH --output=logs/smoke_%j.out
source lumi/common.sh
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== feas2 smoke"; STEPS=40 B=32 NEV=30 EVAL_EVERY=40 TAG=smoke python feasibility/feas2_lexmem.py
echo "=== feas4 smoke"; STEPS=40 B=16 K=3 NEV=20 EVAL_EVERY=40 TAG=smoke INIT=data/lexmem_base.pt python feasibility/feas4_latent.py
echo "=== SMOKE DONE"
