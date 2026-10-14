"""Where does the net lose against the teacher? Held-out soft-labelled positions (data/games_{TEST}.pkl, SIMMING_BOT 10 s with
its top-16 candidates and simulated win %): run the net, take its argmax, and price it with the teacher's own numbers.
cost = teacher's best win % - win % of the net's move (when the net's move is among the teacher's candidates; otherwise the cost is
at least best - worst listed candidate, reported separately). Positions decided without a sim (endgame / pre-endgame) have only
the teacher's move, so they give agreement but no cost. Teacher exchanges count as disagreements (the net may not exchange).
Usage: CKPT=data/policy_simsoft_scratch_nolex.pt TEST=simsoft_31 EXCH=0 python diag_cost.py"""
import os, pickle, collections, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
E = os.environ.get; torch.set_num_threads(int(E("THREADS", 8)))
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); EXCH = E("EXCH", "0") == "1"
model = PolicyNet(lex); model.load_state_dict(torch.load(E("CKPT", "data/policy_simsoft_scratch_nolex.pt"), map_location="cpu"), strict=False); model.eval()
games = pickle.load(open(f"data/games_{E('TEST', 'simsoft_31')}.pkl", "rb"))
recs = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti, r in enumerate(tr) if r.get("known") or str(r.get("teacher_raw", "")).startswith("(exch")]
helper = MoveDataset(lex, [], exch=EXCH)
class D(Dataset):
    def __len__(self): return len(recs)
    def __getitem__(self, i):
        gi, ti = recs[i]; it = helper.build(games[gi][0][ti], 0.0)
        return None if it is None else (i, it)
def coll(batch):
    batch = [b for b in batch if b is not None]
    return [b[0] for b in batch], [b[1]["_moves"] for b in batch], collate_moves([b[1] for b in batch])
def phase(r):
    u = int(r["unseen"].sum())
    return "endgame (bag empty)" if u <= 7 else "pre-endgame (1-7 in bag)" if u <= 14 else "late (8-30 in bag)" if r["bag"] <= 30 else "middle (31-60)" if r["bag"] <= 60 else "opening (61+)"
rows = []
with torch.no_grad():
    for idx, moves, b in DataLoader(D(), batch_size=32, num_workers=int(E("NWORK", 16)), collate_fn=coll):
        pl, _ = model(b); ks = pl.float().argmax(1).tolist()
        for i, mv, k in zip(idx, moves, ks):
            gi, ti = recs[i]; r = games[gi][0][ti]; m = mv[k]
            net_exch = m[3] == 2; key = None if net_exch else (m[1], m[2], m[3], m[4], m[5])
            raw = str(r.get("teacher_raw", "")); t_exch = raw.startswith("(exch")
            row = dict(phase=phase(r), t_exch=t_exch, net_exch=net_exch, score_gap=None)
            if t_exch: row.update(agree=net_exch, cost=None)
            else:
                t = r["move"]; tkey = (t[1], t[2], t[3], t[4], lex.WIDX.get("".join(LET[r["board"][t[1] + ((0, 1), (1, 0))[t[3]][0] * j, t[2] + ((0, 1), (1, 0))[t[3]][1] * j]] if ch == "." else ch.upper() for j, ch in enumerate(t[5]))))
                row["agree"] = key == tkey
                c = r.get("cands", []); est = r.get("cand_est", [])
                if len(c) > 1:   # simmed position: price the net's move
                    best = max(est)
                    if key in c: row["cost"] = best - est[c.index(key)]; row["listed"] = True
                    else: row["cost"] = best - min(est); row["listed"] = False
                else: row["cost"] = None
                row["score_gap"] = t[6] - m[0] if not net_exch else None
            rows.append(row)
print(f"{len(rows)} held-out positions (games_{E('TEST', 'simsoft_31')}), net {os.path.basename(E('CKPT', 'data/policy_simsoft_scratch_nolex.pt'))}\n")
print(f"{'phase':26s} {'n':>6s} {'agree':>6s} {'priced':>7s} {'mean cost (win %)':>18s} {'cost>=2':>8s} {'cost>=5':>8s} {'unlisted':>9s} {'share of total cost':>20s}")
tot = sum(r["cost"] for r in rows if r.get("cost") is not None)
for ph in ["opening (61+)", "middle (31-60)", "late (8-30 in bag)", "pre-endgame (1-7 in bag)", "endgame (bag empty)"]:
    R = [r for r in rows if r["phase"] == ph]; P = [r for r in R if r.get("cost") is not None]
    if not R: continue
    cs = np.array([r["cost"] for r in P]) if P else np.zeros(0)
    print(f"{ph:26s} {len(R):6d} {np.mean([r['agree'] for r in R]):6.3f} {len(P):7d} {cs.mean() if len(cs) else float('nan'):18.2f} "
          f"{np.mean(cs >= 2) if len(cs) else float('nan'):8.3f} {np.mean(cs >= 5) if len(cs) else float('nan'):8.3f} "
          f"{np.mean([not r['listed'] for r in P]) if P else float('nan'):9.3f} {cs.sum() / tot if len(cs) else 0:20.3f}")
X = [r for r in rows if r["t_exch"]]
print(f"\nteacher exchanged in {len(X)} positions ({len(X)/len(rows):.1%}); net also exchanged in {np.mean([r['net_exch'] for r in X]) if X else 0:.1%}; "
      f"net exchanged where the teacher did not: {sum(r['net_exch'] and not r['t_exch'] for r in rows)}")
P = [r for r in rows if r.get("cost") is not None]
mpg = len(rows) / len(games)
print(f"priced positions: mean cost {np.mean([r['cost'] for r in P]):.2f} win-% points per move; ~{mpg:.1f} positions per game (both sides) "
      f"-> ~{np.mean([r['cost'] for r in P]) * mpg / 2:.1f} win-% points per game for one player (upper-biased: the teacher's max is a noisy max)")
G = [r for r in rows if r["score_gap"] is not None and not r["agree"]]
print(f"disagreements: net's move scores {np.mean([-r['score_gap'] for r in G]):+.1f} points vs the teacher's on average (positive = net greedier)")
