"""Loss / eval curves of the imitation runs: imit2 (from scratch, 30k steps) continued by imit3 (4-ply value target, 20k steps)."""
import re, sys, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1a1a19", "#5f5e58", "#e6e5e0", "#fcfcfb"
def parse(path, offset):
    steps, loss, ev = [], [], []
    for line in open(path):
        m = re.match(r"step (\d+) t=\d+s loss=([\d.]+)", line)
        if m: steps.append(int(m.group(1)) + offset); loss.append(float(m.group(2))); continue
        m = re.match(r"\s*\[(untrained|step \d+)\] hasty-move top1=([\d.]+) top3=([\d.]+) value MAE=([\d.]+) pts(?: \| 4-ply spread MAE=([\d.]+))?", line)
        if m:
            st = 0 if m.group(1) == "untrained" else int(m.group(1)[5:])
            ev.append((st + offset, float(m.group(2)), float(m.group(4)), float(m.group(5)) if m.group(5) else np.nan))
    return np.array(steps), np.array(loss), np.array(ev)
runs = {}
for v, col in (("nolex", BLUE), ("lex", ORANGE)):
    s2, l2, e2 = parse(glob.glob(f"data/logs/imitate_imit2-{v}_*.out")[0], 0)
    s3, l3, e3 = parse(glob.glob(f"data/logs/imitate_imit3-{v}_*.out")[0], 30000)
    runs[v] = (col, np.concatenate([s2, s3]), np.concatenate([l2, l3]), np.concatenate([e2, e3[1:]]))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titleweight": "medium", "axes.titlesize": 11, "axes.titlelocation": "left"})
fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("training step (imitation 0–30k, then continued with the 4-ply value target)")
    ax.axvline(30000, color=INK2, lw=1, ls=":")
def lab(ax, x, y, col, text, dy=0): ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=9)
from matplotlib.lines import Line2D
HANDLES = [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=ORANGE, lw=2)]; NAMES = ["no lexicon", "lexicon branch"]
ax = axs[0, 0]
for v, (col, s, l, e) in runs.items():
    ax.plot(s, l, color=col, lw=2); lab(ax, s[-1], l[-1], col, f"{v} {l[-1]:.2f}", dy=6 if v == "nolex" else -6)
ax.set_title("Training loss (policy cross-entropy + value terms)"); ax.set_ylabel("loss"); ax.set_ylim(0, 3); ax.legend(HANDLES, NAMES, frameon=False)
ax.text(30300, 2.8, "loss terms change here", color=INK2, fontsize=8)
ax = axs[0, 1]
for v, (col, s, l, e) in runs.items():
    ax.plot(e[:, 0], e[:, 1], color=col, lw=2, marker="o", ms=4, mec=SURF, mew=1.2); lab(ax, e[-1, 0], e[-1, 1], col, f"{v} {e[-1,1]:.2f}")
ax.set_ylim(0.5, 0.9); ax.set_title("Top-1 agreement with HastyBot's move (held-out games)"); ax.set_ylabel("fraction"); ax.legend(HANDLES, NAMES, frameon=False, loc="lower right")
ax = axs[1, 0]
for v, (col, s, l, e) in runs.items():
    ax.plot(e[:, 0], e[:, 2], color=col, lw=2, marker="o", ms=4, mec=SURF, mew=1.2); lab(ax, e[-1, 0], e[-1, 2], col, f"{v} {e[-1,2]:.0f}", dy=6 if v == "nolex" else -6)
ax.set_ylim(40, 80); ax.set_title("Final-spread prediction error (held-out games)"); ax.set_ylabel("mean abs. error, points")
ax = axs[1, 1]
for v, (col, s, l, e) in runs.items():
    m = ~np.isnan(e[:, 3]); ax.plot(e[m, 0], e[m, 3], color=col, lw=2, marker="o", ms=4, mec=SURF, mew=1.2); lab(ax, e[m][-1, 0], e[m][-1, 3], col, f"{v} {e[m][-1,3]:.1f}", dy=6 if v == "nolex" else -6)
ax.axhline(33.1, color=INK2, lw=1, ls="--"); ax.text(30500, 33.6, "predict-zero baseline 33.1", color=INK2, fontsize=8)
ax.set_xlim(29000, 51000); ax.set_ylim(25, 40); ax.set_title("4-ply spread-change error (the value the lookahead uses)"); ax.set_ylabel("mean abs. error, points")
fig.suptitle("Imitation runs on LUMI, 2026-09-27/28: no-lexicon vs lexicon-branch nets", x=0.02, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(sys.argv[1], dpi=150, facecolor=SURF); print("saved", sys.argv[1])
