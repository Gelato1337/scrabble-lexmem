"""README figures -> docs/figures/*.png. Training curves are parsed from imitate.py logs (copy them from LUMI logs/ into
data/curves/); the benchmark numbers are the ones in docs/benchmarks.md.
Usage: python tools/make_figures.py"""
import re, os, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "docs/figures"; os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
def save(fig, name): fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=150); plt.close(fig); print(f"wrote {OUT}/{name}.png")

# 1. distillation training curves: top-1 agreement with the teacher's move on held-out positions
def curve(path):
    xs, ys = [], []
    for ln in open(path):
        m = re.search(r"\[step (\d+)\] hasty-move top1=([\d.]+)", ln)
        if m: xs.append(int(m.group(1)) / 1000); ys.append(100 * float(m.group(2)))
    return xs, ys
runs = [("data/curves/imitate_simbot-scratch_*.out", "teacher's chosen move, 160k positions", S2),
        ("data/curves/imitate_soft-scratch_*.out", "teacher's full ranking, 730k positions", S1),
        ("data/curves/imitate_soft-big_*.out", "teacher's full ranking, 2.0M positions", S3)]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for pat, lab, col in runs:
    fs = glob.glob(pat)
    if not fs: continue
    x, y = curve(fs[0]); ax.plot(x, y, color=col, lw=2, marker="o", ms=3.5, label=lab)
    dy = {S1: 10, S2: -8, S3: 0}[col]
    ax.annotate(f"{y[-1]:.1f} %", (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.axhline(87.2, color=REF, lw=1.2, ls="--"); ax.text(1, 85.6, "teacher agrees with itself: 87.2 %", color=INK2, fontsize=9, va="top")
ax.set_xlabel("training step (thousands, batch 48)"); ax.set_ylabel("held-out agreement with the teacher (%)")
ax.set_title("Distilling SIMMING_BOT (10 s): top-1 agreement on held-out positions", loc="left", fontsize=11)
ax.set_ylim(40, 92); ax.legend(frameon=False, loc="lower right", fontsize=9); save(fig, "distillation_curves")

# 2. win-rate ladder vs HastyBot (complete games; +- standard error)
rows = [("Score-greedy Python bot", 35.0, 4.8, "100 games"), ("Copy HastyBot's moves", 51.2, 1.1, "2000"),
        ("Copy human annotators", 48.5, 1.1, "2000"), ("Copy the teacher's chosen move", 50.3, 1.1, "2000"),
        ("Teacher's full ranking (soft targets)", 53.2, 0.8, "4000"), ("Self-play v2, best gated round", 52.5, 0.8, "4000"),
        ("Net + Macondo for the last 7 bag tiles", 58.1, 0.8, "4000")]
fig, ax = plt.subplots(figsize=(8.2, 4.2)); y = list(range(len(rows)))[::-1]
ax.barh(y, [r[1] - 30 for r in rows], left=30, height=0.55, color=S1, edgecolor=SURFACE, linewidth=2)
ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
for yy, r in zip(y, rows): ax.text(r[1] + r[2] + 0.4, yy, f"{r[1]:.1f} %", va="center", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(30, 66)
for xv, lab, ha, dx in [(50, "HastyBot", "center", 0), (59.5, "BestBot\n(published)", "right", -0.3), (62.6, "teacher\n(10 s sim)", "left", 0.3)]:
    ax.axvline(xv, color=REF, lw=1.2, ls="--"); ax.text(xv + dx, len(rows) - 0.25, lab, ha=ha, va="bottom", fontsize=8.5, color=INK2)
ax.set_ylim(-0.6, len(rows) + 0.5)
ax.set_xlabel("win rate vs HastyBot (%)"); ax.grid(axis="y", visible=False)
ax.set_title("Win rate vs HastyBot, complete games", loc="left", fontsize=11, pad=22); save(fig, "win_ladder")

# 3. where games are lost: share of positions vs share of the net's total cost (teacher win %), by phase
ph = ["opening\n(61+ in bag)", "middle\n(31-60)", "late\n(8-30)", "pre-endgame\n(1-7)"]
pos = [26.8, 29.3, 24.3, 7.6]; cost = [14.9, 17.8, 26.3, 41.1]
fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = range(len(ph)); w = 0.36
b1 = ax.bar([i - w / 2 for i in x], pos, w, color=S1, edgecolor=SURFACE, linewidth=2, label="share of positions")
b2 = ax.bar([i + w / 2 for i in x], cost, w, color=S2, edgecolor=SURFACE, linewidth=2, label="share of the win % lost to the teacher")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f} %", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(list(x)); ax.set_xticklabels(ph); ax.set_ylabel("% of total"); ax.set_ylim(0, 48); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Where the net loses against the teacher, by game phase", loc="left", fontsize=11)
save(fig, "cost_by_phase")

# 4. net + Macondo at the end of the game
labs = ["net alone", "+ solver,\nbag empty", "+ solver,\n<=1 tile in bag", "+ teacher bot,\n<=7 tiles in bag", "teacher bot,\nwhole game"]
vals = [53.2, 54.9, 55.2, 58.1, 62.6]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
bars = ax.bar(range(5), [v - 45 for v in vals], bottom=45, width=0.55, color=[S1] * 4 + [REF], edgecolor=SURFACE, linewidth=2)
for i, v in enumerate(vals): ax.text(i, v + 0.4, f"{v:.1f} %", ha="center", fontsize=9, color=INK)
ax.axhline(50, color=REF, lw=1.2, ls="--"); ax.text(-0.42, 50.3, "HastyBot", ha="left", fontsize=8.5, color=INK2)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=9); ax.set_ylim(45, 65); ax.set_ylabel("win rate vs HastyBot (%)"); ax.grid(axis="x", visible=False)
ax.set_title("Handing the end of the game to Macondo", loc="left", fontsize=11)
save(fig, "endgame_hybrid")

# 5. lexicon memory, sample efficiency: held-out value quality vs training positions (rollout targets, frozen query encoder; 2 seeds)
N = [10000, 50000, 177213]
r2 = {"no lexicon memory": [[.124, .134], [.232, .231], [.279, .252]], "lexicon memory": [[.117, .117], [.243, .251], [.251, .262]]}
ll = {"no lexicon memory": [[.4499, .4487], [.4391, .4391], [.4315, .4338]], "lexicon memory": [[.4502, .4467], [.4369, .4386], [.4340, .4332]]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, data, lab in ((axes[0], r2, "held-out R² of the score difference (higher is better)"), (axes[1], ll, "held-out win log-loss (lower is better)")):
    for (name, vals), col, dx in zip(data.items(), (S2, S1), (0.97, 1.03)):
        m = [sum(v) / 2 for v in vals]
        ax.plot([n * dx for n in N], m, color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * 2, v, color=col, s=12, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["10k", "50k", "177k"]); ax.set_xlabel("training positions"); ax.set_title(lab, loc="left", fontsize=9.5)
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Lexicon memory vs none: value from our own rollouts (19,703 held-out positions; dots = seeds)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexmem_sample_efficiency")
