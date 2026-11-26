#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=bignet-games
#SBATCH --output=logs/bignet_games_%j.out
# The reference policy net and the bigger one, each alone (one forward pass per move) against the current HastyBot: NG games each
# on seeds 777000 + g, so the two runs pair game by game. CPU only. Env: REF NEW (checkpoint names without policy_ / .pt) NG.
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
set -o pipefail
REF=${REF:-simsoft_scratch_nolex}; NEW=${NEW:-simsoft_scratch_nolex_d512l6}; NG=${NG:-4000}
for c in $REF $NEW; do CKPT=data/policy_$c.pt USE_LEX=0 NPROC=60 TAGE=_hfix python eval/play_eval.py $NG 777000 | tail -1 || exit 1; done
python - <<PY || exit 1
import pickle, numpy as np
a, b = (pickle.load(open(f"data/eval_policy_{c}_hfix_$NG.pkl", "rb")) for c in ("$REF", "$NEW"))
se = lambda x: x.std(ddof=1) / np.sqrt(len(x)); wa, wb = 100.0 * (a > 0), 100.0 * (b > 0)
print(f"$REF: win-% {wa.mean():.1f} +- {se(wa):.1f}, spread {a.mean():+.1f}")
print(f"$NEW: win-% {wb.mean():.1f} +- {se(wb):.1f}, spread {b.mean():+.1f} | paired vs $REF: win-% {(wb - wa).mean():+.1f} +- {se(wb - wa):.1f}, spread {(b - a).mean():+.1f} +- {se(b - a):.1f}")
PY
echo "=== BIGNET GAMES DONE"
