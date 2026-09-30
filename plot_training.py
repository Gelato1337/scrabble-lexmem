"""Training-curve figure from LUMI logs. Usage: python plot_training.py out.png"""
import re, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1a1a19", "#5f5e58", "#e6e5e0", "#fcfcfb"
def parse(path):
    steps, loss, ev = [], [], []
    for line in open(path):
        m = re.match(r"step (\d+) t=(\d+)s loss=([\d.]+)", line)
        if m: steps.append(int(m.group(1))); loss.append(float(m.group(3))); continue
        m = re.match(r"\s*\[(untrained|step \d+)\] hasty-move top1=([\d.]+) top3=([\d.]+) value MAE=([\d.]+)", line)
        if m: ev.append((0 if m.group(1) == "untrained" else int(m.group(1)[5:]), float(m.group(2)), float(m.group(3)), float(m.group(4)))); continue
        m = re.match(r"\s*\[real\] ndcg=([\d.]+)", line)
        if m and steps: ev.append((steps[-1], float(m.group(1))))
    return np.array(steps), np.array(loss), ev
imit_s, imit_l, imit_ev = parse("data/logs/imitate_imit-nolex_22376045.out")
ret_s, ret_l, ret_ev = parse("data/logs/feas2_nwl_22376033.out")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titleweight": "medium", "axes.titlesize": 11, "axes.titlelocation": "left"})
fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("training step")
def style_end(ax, x, y, color, label):
    ax.plot(x[-1], y[-1], "o", ms=5, color=color, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
# 1 imitation loss
ax = axs[0, 0]; ax.plot(imit_s, imit_l, color=BLUE, lw=2); style_end(ax, imit_s, imit_l, BLUE, f"{imit_l[-1]:.2f}")
ax.set_title("Imitation net (no lexicon branch): training loss"); ax.set_ylabel("cross-entropy + 0.5·value loss")
ax.axhline(np.log(400), color=INK2, lw=1, ls=":"); ax.text(imit_s[0], np.log(400) + 0.08, "random over ~400 moves", color=INK2, fontsize=8)
# 2 move agreement
ax = axs[0, 1]
if imit_ev:
    e = np.array([x for x in imit_ev if len(x) == 4]); ax.plot(e[:, 0], e[:, 1], color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5)
    ax.plot(e[:, 0], e[:, 2], color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5)
    ax.annotate(f"top-1  {e[-1,1]:.2f}", (e[-1, 0], e[-1, 1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.annotate(f"top-3  {e[-1,2]:.2f}", (e[-1, 0], e[-1, 2]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.legend(["top-1", "top-3"], frameon=False, loc="lower right")
ax.set_xlim(0, imit_s[-1] * 1.08); ax.set_ylim(0, 1); ax.set_title("Agreement with HastyBot's move (held-out games)"); ax.set_ylabel("fraction of positions")
# 3 value MAE
ax = axs[1, 0]
if imit_ev:
    e = np.array([x for x in imit_ev if len(x) == 4]); ax.plot(e[:, 0], e[:, 3], color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5)
    ax.annotate(f"{e[-1,3]:.0f} pts", (e[-1, 0], e[-1, 3]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xlim(0, imit_s[-1] * 1.08); ax.set_ylim(0, 100); ax.set_title("Final-spread prediction error (held-out games)"); ax.set_ylabel("mean abs. error, points")
# 4 retrieval loss + ndcg
ax = axs[1, 1]; ax.plot(ret_s, ret_l, color=BLUE, lw=2); style_end(ax, ret_s, ret_l, BLUE, f"{ret_l[-1]:.2f}")
ax.set_title("Lexicon layer on NWL23 (196k words): training loss"); ax.set_ylabel("listwise loss")
if ret_ev:
    e = np.array([x for x in ret_ev if len(x) == 2])
    for s, nd in e: ax.annotate(f"NDCG {nd:.2f}", (s, np.interp(s, ret_s, ret_l)), xytext=(0, -14), textcoords="offset points", ha="center", color=ORANGE, fontsize=8)
fig.suptitle(f"LUMI runs 2026-09-27 · imitation at step {imit_s[-1]} of 30000 · retrieval at step {ret_s[-1]} of 40000", x=0.02, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(sys.argv[1], dpi=150, facecolor=SURF); print("saved", sys.argv[1])
