#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweep-smoke
#SBATCH --output=logs/sweep_smoke_%j.out
# CPU check of train/imitate_sweep.py: a net trained by it must equal the net train/imitate.py trains with the same settings
# (40 steps of 4); then eval/policy_agree.py and lumi/bignet_games.sh (8 games, several nets) on the small nets. No GPU.
# Every python call stops itself with a traceback of all threads if it has not ended in time.
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
set -o pipefail
WD() { python -c "import faulthandler, runpy, sys; faulthandler.dump_traceback_later($1, exit=True); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" "${@:2}"; }
C="TRAIN=simsoft_0 TEST=simsoft_31 LEXICON=data/NWL23.txt TAU=1.0 MARGIN=0 STEPS=40 B=4 NWORK=8 NEV=16 EVAL_EVERY=20"
date +"%T single"
( export $C USE_LEX=0 EI=1 BETA=0 PLAY=0 D=256 LAYERS=3 LR=1e-4 WARMUP=20 TAG=sweep_smoke_single; WD 420 train/imitate.py ) || echo "!!! train/imitate.py did not end cleanly"
date +"%T sweep"
env $C STALL=300 SWEEP="sweep_smoke_a:256:3:1e-4:20 sweep_smoke_b:384:4:2e-4:10 sweep_smoke_c:256:3:1e-4:20" python train/imitate_sweep.py || exit 1
date +"%T compare"
python - <<'PY' || exit 1
import torch
s, a, b, c = (torch.load(f"data/policy_sweep_smoke_{t}.pt", map_location="cpu") for t in ("single", "a", "b", "c"))
d = lambda x, y: max(float((x[k].float() - y[k].float()).abs().max()) for k in x)
print("WEIGHTS single vs a: max abs difference", d(s, a), "| a vs c:", d(a, c), "| same keys:", s.keys() == a.keys(), "| b width", b["cls"].shape[-1])
PY
( export TEST=simsoft_31 N=64 NWORK=8; WD 420 eval/policy_agree.py data/policy_sweep_smoke_single.pt data/policy_sweep_smoke_a.pt data/policy_sweep_smoke_b.pt ) 2>&1 | grep -v "Warning\|warnings.warn" || exit 1
rm -f data/eval_policy_sweep_smoke_?_hfix_8.pkl
REF=sweep_smoke_a NEW="sweep_smoke_b sweep_smoke_c" NG=8 NPROC=8 bash lumi/bignet_games.sh 2>&1 | grep -v "Warning\|warnings.warn\|Singularity\|python3, pip3\|installed by CSC\|docs.csc.fi\|^$" || exit 1
REF=sweep_smoke_a NEW="sweep_smoke_b sweep_smoke_c" NG=8 NPROC=8 bash lumi/bignet_games.sh 2>&1 | grep -v "Warning\|warnings.warn\|Singularity\|python3, pip3\|installed by CSC\|docs.csc.fi\|^$" || exit 1
date +"%T device_count on this node"
python -c "import faulthandler; faulthandler.dump_traceback_later(90, exit=True); import torch; print('device_count', torch.cuda.device_count())" || echo "!!! torch.cuda.device_count() did not return on this node"
echo "=== SMOKE DONE"
