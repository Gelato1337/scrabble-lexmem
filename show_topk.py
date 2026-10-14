"""Render held-out mid-game positions with the policy net's top-5 moves, next to HastyBot's and SIMMING_BOT's choices.
Positions: data/games_{SRC}.pkl (default hasty_test, never trained on), bag 30-60 tiles, the mover's actual move a play.
Usage: CKPT=data/policy_imit3_nolex.pt SRC=hasty_test NPOS=6 SECS=10 python show_topk.py docs/samples
Writes <out>/position_<i>.png and <out>/positions.md (the same numbers as text)."""
import os, sys, random, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
from harness import MacondoBot
E = os.environ.get; torch.set_num_threads(4)
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/samples"; os.makedirs(OUT, exist_ok=True)
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); CKPT = E("CKPT", "data/policy_imit3_nolex.pt")
model = PolicyNet(lex, use_lex=False); model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False); model.eval()
helper = MoveDataset(lex, [])
import pickle; games = pickle.load(open(f"data/games_{E('SRC', 'hasty_test')}.pkl", "rb"))
rng = random.Random(int(E("SEED", 3))); pool = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti, r in enumerate(tr) if 30 <= r["bag"] <= 60 and r["move"][0] == "play"]
picks = rng.sample(pool, int(E("NPOS", 6)))
os.environ["CANDS"] = "5"; sim = MacondoBot("SIMMING_BOT", float(E("SECS", 10))); hasty = MacondoBot("HASTY_BOT", 1)

def coord(r, c, d): return f"{r+1}{LET[c]}" if d == 0 else f"{LET[c]}{r+1}"
def rack_str(rack): return "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(int(rack[t])))
def move_letters(board, rack, r, c, d, L, w):
    """-> display word with (through tiles), placed squares [(r,c,ch)], leave string."""
    dr, dc = ((0, 1), (1, 0))[d]; tmp = rack.copy(); disp = ""; placed = []; inside = False
    for j in range(L):
        rr, cc = r + dr * j, c + dc * j; ch = LET[lex.POS[w, j]]
        if board[rr, cc] >= 0:
            disp += ("" if inside else "(") + ch; inside = True; continue
        if inside: disp += ")"; inside = False
        if tmp[IDX[ch]] > 0: tmp[IDX[ch]] -= 1
        else: tmp[BLANK] -= 1; ch = ch.lower()
        disp += ch; placed.append((rr, cc, ch))
    if inside: disp += ")"
    return disp, placed, rack_str(tmp)

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
PREMC = {(1, 3): "#e8605a", (1, 2): "#f6b9b5", (3, 1): "#5b8fd1", (2, 1): "#bcd6f2"}   # (letter mult, word mult): TW red, DW pink, TL blue, DL light blue
md = [f"# Policy net top-5 on held-out mid-game positions\n\nCheckpoint `{os.path.basename(CKPT)}`; positions from `games_{E('SRC', 'hasty_test')}` "
      f"(HastyBot self-play, never trained on), 30-60 tiles in the bag. SIMMING_BOT = Macondo's sim bot, {E('SECS', 10)} s, one thread; "
      "win % is its simulated win probability for that move.\n"]
for k, (gi, ti) in enumerate(picks):
    rec = games[gi][0][ti]; board, blanks, rack = rec["board"], rec["blanks"], rec["rack"]
    it = helper.build(rec, 0.0); b = collate_moves([it])
    with torch.no_grad(): pl, v = model(b)
    p = torch.softmax(pl[0].float(), 0).numpy(); order = np.argsort(-p)[:5]
    top = []
    for i in order:
        s, r, c, d, L, w = it["_moves"][i]; disp, placed, leave = move_letters(board, rack, r, c, d, L, w)
        top.append(dict(coord=coord(r, c, d), word=disp, score=s, leave=leave, p=p[i], placed=placed))
    zeros = 0
    h_play = hasty.move(board, blanks, rack, tuple(int(x) for x in rec["scores"]), zeros)
    s_play = sim.move(board, blanks, rack, tuple(int(x) for x in rec["scores"]), zeros); s_cands = sim.last_cands
    # --- figure
    fig = plt.figure(figsize=(13, 7.2)); ax = fig.add_axes([0.02, 0.07, 0.5, 0.88]); ax.set_xlim(0, 15); ax.set_ylim(15, 0); ax.set_aspect("equal"); ax.axis("off")
    for r in range(15):
        for c in range(15):
            col = PREMC.get((int(LMULT[r, c]), int(WMULT[r, c])), "#f5f0e1")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=col, edgecolor="#999", lw=0.5))
            if board[r, c] >= 0:
                ax.add_patch(Rectangle((c + 0.05, r + 0.05), 0.9, 0.9, facecolor="#e8c98f", edgecolor="#8a6d3b", lw=0.8))
                ch = LET[board[r, c]]; ax.text(c + 0.5, r + 0.55, ch.lower() if blanks[r, c] else ch, ha="center", va="center", fontsize=11, weight="bold")
    for rank in range(len(top) - 1, -1, -1):   # draw #1 last so it is on top
        t = top[rank]
        for (rr, cc, ch) in t["placed"]:
            off = 0.06 * rank
            ax.add_patch(Rectangle((cc + 0.08 + off / 2, rr + 0.08 + off / 2), 0.84 - off, 0.84 - off, fill=rank == 0, facecolor="#fff7d6", edgecolor=COLORS[rank], lw=2.2))
            if rank == 0: ax.text(cc + 0.5, rr + 0.55, ch, ha="center", va="center", fontsize=11, weight="bold", color=COLORS[0])
    for i in range(15):
        ax.text(i + 0.5, -0.3, LET[i], ha="center", fontsize=8, color="#555"); ax.text(-0.3, i + 0.55, str(i + 1), ha="right", va="center", fontsize=8, color="#555")
    for i, (lab, col) in enumerate([("TW", "#e8605a"), ("DW", "#f6b9b5"), ("TL", "#5b8fd1"), ("DL", "#bcd6f2")]):
        ax.add_patch(Rectangle((i * 2.2, 15.25), 0.5, 0.5, facecolor=col, edgecolor="#999", clip_on=False)); ax.text(i * 2.2 + 0.65, 15.55, lab, fontsize=8, va="center")
    ax.text(9.0, 15.55, "outlined squares = net's top 5 (colour = rank)", fontsize=8, va="center")
    tx = fig.add_axes([0.54, 0.04, 0.45, 0.9]); tx.axis("off")
    lines = [(f"Position {k+1}   (game {gi}, turn {ti})", 13, "black", "bold"),
             (f"Rack {rack_str(rack)}   score {rec['scores'][0]}-{rec['scores'][1]}   bag {rec['bag']}", 11, "black", "normal"),
             (f"Net value estimate (final spread, mover): {float(v[0])*100:+.0f}", 10, "#444", "normal"), ("", 6, "black", "normal"),
             ("Net's top 5            score  leave      prob", 11, "black", "bold")]
    for rank, t in enumerate(top):
        lines.append((f"{rank+1}. {t['coord']:>4} {t['word']:<12} {t['score']:>4}  {t['leave']:<8} {t['p']*100:5.1f}%", 11, COLORS[rank], "normal"))
    lines += [("", 6, "black", "normal"), (f"HastyBot plays:     {h_play}", 11, "black", "normal"), (f"SIMMING_BOT plays:  {s_play}", 11, "black", "bold"),
              ("SIMMING_BOT top 5 (simulated win %)", 10, "#444", "normal")]
    for wp, eq, n, ign, pl_ in s_cands[:5]: lines.append((f"   {pl_:<18} {wp*100:5.1f}%  ({n} iters)", 10, "#444", "normal"))
    y = 0.98
    for txt, fs, col, wt in lines: tx.text(0.0, y, txt, fontsize=fs, color=col, weight=wt, family="monospace", va="top"); y -= 0.028 * fs / 10 + 0.012
    fig.savefig(f"{OUT}/position_{k+1}.png", dpi=110); plt.close(fig)
    md.append(f"## Position {k+1}\n\n![position {k+1}](position_{k+1}.png)\n\nRack `{rack_str(rack)}`, score {rec['scores'][0]}-{rec['scores'][1]}, bag {rec['bag']}.\n\n"
              "| # | move | score | leave | net prob |\n|---|---|---|---|---|\n" +
              "".join(f"| {i+1} | {t['coord']} {t['word']} | {t['score']} | {t['leave']} | {t['p']*100:.1f} % |\n" for i, t in enumerate(top)) +
              f"\nHastyBot: `{h_play}` · SIMMING_BOT: `{s_play}` (" + ", ".join(f"{pl_} {wp*100:.1f} %" for wp, _, _, _, pl_ in s_cands[:5]) + ")\n")
    print(f"position {k+1}: net {top[0]['coord']} {top[0]['word']} ({top[0]['p']*100:.0f}%) | hasty {h_play} | sim {s_play}", flush=True)
open(f"{OUT}/positions.md", "w").write("\n".join(md)); sim.close(); hasty.close()
