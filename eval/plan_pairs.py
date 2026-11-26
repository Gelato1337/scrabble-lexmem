"""Planning benchmark: runs compared position by position.
Input: data/planrows_<TAG>.pkl (train/train_lexplan.py: the last evaluation's rows per held-out position; EVAL_CKPTS writes them
for old checkpoints). An argument is one tag or several tags joined by commas (seeds of one arm: their regret and accuracy are
averaged per position first).
Usage: python eval/plan_pairs.py BASE ARM [ARM ...]
Output per arm: decision accuracy and regret (points) on all positions and on the planning-critical ones, and the paired
difference to BASE (mean over positions +- standard error; negative regret difference = the arm is better)."""
import sys, pickle, numpy as np

def load(arg):
    runs = [pickle.load(open(f"data/planrows_{t}.pkl", "rb")) for t in arg.split(",")]
    pos = runs[0]["pos"]
    for r in runs[1:]: assert (r["pos"] == pos).all(), f"{arg}: seeds evaluated on different positions"
    R = np.mean([r["rows"] for r in runs], 0)
    return {int(p): R[k] for k, p in enumerate(pos)}
def pm(x): return f"{x.mean():+.3f} +- {x.std(ddof=1) / np.sqrt(len(x)):.3f}"

base = load(sys.argv[1])
for arg in sys.argv[1:]:
    A = load(arg); common = sorted(set(base) & set(A)); a = np.array([A[p] for p in common]); b = np.array([base[p] for p in common]); crit = b[:, 4] == 1
    line = f"{arg}\n    {len(common)} positions: accuracy {a[:, 0].mean():.3f}, regret {a[:, 1].mean():.3f} | critical n={int(crit.sum())}: accuracy {a[crit, 0].mean():.3f}, regret {a[crit, 1].mean():.3f}"
    if arg != sys.argv[1]:
        line += f"\n    vs {sys.argv[1]}: regret all {pm(a[:, 1] - b[:, 1])}, critical {pm(a[crit, 1] - b[crit, 1])}; accuracy all {pm(a[:, 0] - b[:, 0])}, critical {pm(a[crit, 0] - b[crit, 0])}"
    print(line)
