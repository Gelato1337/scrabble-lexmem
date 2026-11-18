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

# 6. planning benchmark: within-position R² (what decides moves) of the two lexicon-dependent quantities vs training positions
N = [1000, 3000, 10000, 40000, 270000]
wo = {"no lexicon memory": [[-.064, -.043, -.059], [-.033, -.032, -.108], [.021, -.016, .054], [.059, .139], [.266, .264]],
      "lexicon memory": [[.139, .128, .139], [.181, .134, .163], [.224, .178, .223], [.171, .241], [.283, .304]]}
creg = {"no lexicon memory": [[5.38, 6.35, 4.76], [5.96, 4.83, 5.08], [4.51, 4.75, 4.76], [4.63, 4.21], [3.95, 4.16]],
        "lexicon memory": [[5.06, 5.51, 5.06], [4.51, 4.55, 4.67], [4.54, 4.41, 4.25], [4.20, 4.00], [3.99, 3.78]]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, data, lab, yl in ((axes[0], wo, "ranking candidates by the opponent's best reply", "within-position R² of E_opp"),
                          (axes[1], creg, "decisions on planning-critical positions", "regret vs exact answer (points)")):
    for (name, vals), col, dx in zip(data.items(), (S2, S1), (0.96, 1.04)):
        ax.plot([n * dx for n in N], [sum(v) / len(v) for v in vals], color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * len(v), v, color=col, s=12, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["1k", "3k", "10k", "40k", "270k"]); ax.set_xlabel("training positions")
    ax.set_ylabel(yl); ax.set_title(lab, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
axes[1].text(1000, 10.2, "", fontsize=8)
fig.suptitle("Planning benchmark, lexicon memory vs none (3000 held-out positions; dots = seeds; top-score baseline regret: 10.5)", x=0.01, ha="left", fontsize=10)
save(fig, "lexplan_within_r2")

# 7. planning policies in real games vs HastyBot (4000 games per run, +- 0.8 each; seeds 0 and 1 of each net; exact planner with the
# fixed labeller of 2026-11-17); same nets in both dictionaries
N = [3000, 10000, 270000]
games = (("NWL23 (in-domain)", {"no lexicon memory": [[38.9, 43.1], [42.5, 41.0], [43.4, 44.0]], "lexicon memory": [[43.4, 43.0], [43.7, 43.3], [43.5, 43.2]]}, 48.2, 36.6),
         ("CSW24, zero-shot (HastyBot plays CSW24)", {"no lexicon memory": [[40.0, 42.1], [41.6, 40.9], [45.7, 43.5]], "lexicon memory": [[41.4, 42.5], [43.2, 43.8], [43.7, 44.0]]}, 48.1, 35.6))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (title, wr, ex, ts) in zip(axes, games):
    for (name, vals), col, dx in zip(wr.items(), (S2, S1), (0.97, 1.03)):
        ax.plot([n * dx for n in N], [sum(v) / 2 for v in vals], color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * 2, v, color=col, s=14, alpha=0.5, zorder=3)
    for yv, lab in ((ex, f"exact planner {ex}"), (ts, f"top score, no planning {ts}")):
        ax.axhline(yv, color=REF, lw=1.2, ls="--"); ax.text(2300, yv + 0.35, lab, fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["3k", "10k", "270k"]); ax.set_xlabel("NWL training positions of the planning net")
    ax.set_ylim(34, 50); ax.set_title(title, loc="left", fontsize=10)
axes[0].set_ylabel("win rate vs HastyBot (%)"); axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Planning policy in games (4000 games per run, ± 0.8; dots = seeds 0, 1; line = mean)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexplan_games")

# 8. dictionary transfer: planners trained on NWL23 positions, evaluated on the same positions relabelled under CSW24
N = [1000, 3000, 10000, 40000, 270000]
zs_wo = {"no lexicon memory": [[-.056, -.029, -.048], [-.019, -.029, -.086], [.027, -.015, .051], [.053, .136], [.242, .236]],
         "lexicon memory": [[.152, .137, .138], [.178, .139, .168], [.211, .175, .204], [.186, .238], [.261, .301]]}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for (name, vals), col, dx in zip(zs_wo.items(), (S2, S1), (0.96, 1.04)):
    ax.plot([n * dx for n in N], [sum(v) / len(v) for v in vals], color=col, lw=2, marker="o", ms=5, label=f"{name}, trained on NWL")
    for n, v in zip(N, vals): ax.scatter([n * dx] * len(v), v, color=col, s=12, alpha=0.5, zorder=3)
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["1k", "3k", "10k", "40k", "270k"]); ax.set_xlabel("NWL training positions")
ax.set_ylabel("within-position R² of E_opp on CSW"); ax.set_title("zero-shot on CSW24: ranking by the opponent's threat", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax = axes[1]; conds = ["NWL\n(in-domain)", "CSW\nzero-shot", "CSW\n(trained on CSW)"]
nol = [4.06, 4.43, 4.05]; lx = [3.89, 4.00, 3.81]; x = list(range(3))
for vals, col, name, dx in ((nol, S2, "no lexicon memory", -0.08), (lx, S1, "lexicon memory", 0.08)):
    ax.scatter([i + dx for i in x], vals, color=col, s=55, zorder=3, label=name)
    for i, v in zip(x, vals): ax.text(i + dx + (0.1 if dx > 0 else -0.1), v, f"{v:.2f}", ha="left" if dx > 0 else "right", va="center", fontsize=8.5, color=INK2)
for i in x: ax.plot([i - 0.08, i + 0.08], [nol[i], lx[i]], color=GRID, lw=1.5, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=9); ax.set_xlim(-0.6, 2.6); ax.set_ylim(3.6, 4.6); ax.grid(axis="x", visible=False)
ax.set_ylabel("critical-position regret (points)"); ax.set_title("270k training positions, mean of 2 seeds", loc="left", fontsize=10)
fig.suptitle("Dictionary transfer NWL23 → CSW24 (same 3000 held-out positions; +88k words)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexplan_csw_transfer")

# 9. isolation: lane tokens without cross-check letters (the control's only dictionary-derived input), both arms, zero-shot on CSW
N = [3000, 10000, 270000]
iso = {("no lexicon memory", True): [[-.019, -.029], [.027, -.015], [.242, .236]], ("no lexicon memory", False): [[-.055, -.003], [.006, -.034], [.216, .242]],
       ("lexicon memory", True): [[.178, .139], [.211, .175], [.261, .301]], ("lexicon memory", False): [[.185, .161], [.198, .164], [.272, .304]]}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for (name, xchk), vals in iso.items():
    col = S1 if name == "lexicon memory" else S2; dx = (1.04 if name == "lexicon memory" else 0.96) * (1.0 if xchk else 1.02)
    ax.plot([n * dx for n in N], [sum(v) / len(v) for v in vals], color=col, lw=2, ls="-" if xchk else "--", marker="o" if xchk else "s",
            ms=5, mfc=col if xchk else SURFACE, label=f"{name}, {'with' if xchk else 'without'} cross-checks")
    for n, v in zip(N, vals): ax.scatter([n * dx] * len(v), v, color=col, s=12, alpha=0.5, zorder=3)
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["3k", "10k", "270k"]); ax.set_xlabel("NWL training positions")
ax.set_ylabel("within-position R² of E_opp on CSW"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Removing cross-check sets changes neither arm (zero-shot on CSW24; dots = seeds 0, 1)", loc="left", fontsize=10)
save(fig, "lexplan_no_xchk")

# 10. clean control: a no-memory net that trains its own query encoder (no dictionary-trained part) vs the round-2 arms (frozen
# future-word encoder), all without cross-check sets; planning benchmark, NWL, 3000 held-out positions; dots = seeds 0, 1
N = [3000, 10000, 270000]
arms = {"no memory, frozen encoder (3.5M trained)": (S2, [[-.074, -.01], [.005, -.052], [.247, .257]], [[4.5, 4.63], [4.76, 4.75], [4.09, 4.02]]),
        "lexicon memory, frozen encoder (3.6M)": (S1, [[.191, .159], [.206, .163], [.273, .297]], [[4.6, 4.35], [4.58, 4.32], [4.36, 3.94]]),
        "no memory, encoder trained from scratch (6.4M)": (S3, [[.051, .016], [.106, .084], [.293, .274]], [[4.19, 4.51], [4.35, 4.25], [3.74, 3.94]])}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for k, (ax, yl, ttl) in enumerate(((axes[0], "within-position R² of E_opp", "ranking candidates by the opponent's best reply"),
                                   (axes[1], "critical-position regret (points)", "decisions on planning-critical positions"))):
    for (name, (col, wo, cr)), dx in zip(arms.items(), (0.95, 1.0, 1.05)):
        vals = (wo, cr)[k]
        ax.plot([n * dx for n in N], [sum(v) / 2 for v in vals], color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * 2, v, color=col, s=12, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["3k", "10k", "270k"]); ax.set_xlabel("training positions")
    ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Clean control: no cross-checks, no dictionary-trained part (NWL planning benchmark; top-score regret 10.5)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexplan_clean_control")

# 11. two-turn horizon: labels from go/lexmem_plan H2=1 (fixed labeller), PV2 = score - E_opp + E_my2 - E_opp2; round-2 setting
# (frozen encoder, cross-checks on); 3000 held-out positions; dots = seeds 0, 1
h2 = {"no lexicon memory": (S2, [[.01, .008], [.051, .046], [.281, .306]], [[5.42, 4.87], [4.6, 4.64], [3.4, 3.42]]),
      "lexicon memory": (S1, [[.209, .236], [.236, .278], [.328, .369]], [[4.31, 4.01], [3.88, 3.76], [3.37, 3.17]])}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for k, (ax, yl, ttl) in enumerate(((axes[0], "within-position R² of E_opp", "ranking candidates by the opponent's best reply"),
                                   (axes[1], "critical-position regret on PV2 (points)", "two-turn decisions, planning-critical positions"))):
    for (name, (col, wo, cr)), dx in zip(h2.items(), (0.96, 1.04)):
        vals = (wo, cr)[k]
        ax.plot([n * dx for n in N], [sum(v) / 2 for v in vals], color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * 2, v, color=col, s=12, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["3k", "10k", "270k"]); ax.set_xlabel("training positions")
    ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Two-turn planning benchmark (opponent's reply, my answer, opponent again; top-score regret 10.7)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexplan_two_turn")

# 12. capacity-matched test: every net trains its query encoder (memory 6.44M, no memory 6.37M trainable), no cross-checks;
# one turn = NWL planning benchmark, round-2 labels; two turns = fixed labels, PV2; 3000 held-out positions; dots = seeds 0, 1
cm = ((("one turn (NWL, PV)", "critical-position regret (points)"),
       {"no memory, encoder trained from scratch": (S3, [[4.19, 4.51], [4.35, 4.25], [3.74, 3.94]]),
        "no memory, pretrained encoder trained": (S2, [[4.2, 4.49], [4.65, 4.64], [3.66, 4.57]]),
        "lexicon memory, pretrained encoder trained": (S1, [[4.06, 4.03], [4.46, 4.34], [4.12, 3.78]])}),
      (("two turns (PV2)", "critical-position regret on PV2 (points)"),
       {"no memory, encoder trained from scratch": (S3, [[4.21, 4.39], [3.88, 4.31], [3.49, 3.63]]),
        "lexicon memory, pretrained encoder trained": (S1, [[3.61, 3.87], [3.95, 4.12], [3.17, 3.11]])}))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, ((ttl, yl), arms) in zip(axes, cm):
    for name, (col, vals) in arms.items():
        dx = {S3: 0.95, S2: 1.0, S1: 1.05}[col]
        ax.plot([n * dx for n in N], [sum(v) / 2 for v in vals], color=col, lw=2, marker="o", ms=5, label=name)
        for n, v in zip(N, vals): ax.scatter([n * dx] * 2, v, color=col, s=12, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["3k", "10k", "270k"]); ax.set_xlabel("training positions")
    ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Capacity-matched: all nets train their query encoder, no cross-checks (top-score regret 10.5 / 10.7)", x=0.01, ha="left", fontsize=10.5)
save(fig, "lexplan_capacity_matched")
