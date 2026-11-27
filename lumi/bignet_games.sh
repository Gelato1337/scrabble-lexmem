#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=bignet-games
#SBATCH --output=logs/bignet_games_%j.out
# The reference policy net and other ones, each alone (one forward pass per move) against the current HastyBot: NG games each on
# seeds 777000 + g, so the runs pair game by game. A net whose result file exists is not played again. CPU only.
# Env: REF, NEW (checkpoint names without policy_ / .pt; NEW may hold several, space-separated) NG NPROC.
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
set -o pipefail
REF=${REF:-simsoft_scratch_nolex}; NEW=${NEW:-simsoft_scratch_nolex_d512l6}; NG=${NG:-4000}
for c in $REF $NEW; do
  [ -s data/eval_policy_${c}_hfix_$NG.pkl ] && { echo "$c: result file exists, not played again"; continue; }
  CKPT=data/policy_$c.pt USE_LEX=0 NPROC=${NPROC:-60} TAGE=_hfix python eval/play_eval.py $NG 777000 | tail -1 || exit 1
done
python - <<PY || exit 1
import pickle, numpy as np
se = lambda x: x.std(ddof=1) / np.sqrt(len(x)); a = pickle.load(open("data/eval_policy_${REF}_hfix_$NG.pkl", "rb")); wa = 100.0 * (a > 0)
print(f"$REF: win-% {wa.mean():.1f} +- {se(wa):.1f}, spread {a.mean():+.1f}")
for c in "$NEW".split():
    b = pickle.load(open(f"data/eval_policy_{c}_hfix_$NG.pkl", "rb")); wb = 100.0 * (b > 0)
    print(f"{c}: win-% {wb.mean():.1f} +- {se(wb):.1f}, spread {b.mean():+.1f} | paired vs $REF: win-% {(wb - wa).mean():+.1f} +- {se(wb - wa):.1f}, spread {(b - a).mean():+.1f} +- {se(b - a):.1f}")
PY
echo "=== BIGNET GAMES DONE"
