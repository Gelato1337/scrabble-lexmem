"""Go engine output (one JSON line per game, bin/lexmem_selfplay) -> our trajectory pickle (data/games_{TAG}.pkl), as read by
train_v2.py / train_branch.py. Plays get their letters from designate() (blanks on the best squares, the same score Macondo used);
branches become rec["branch"] = (alternative move tuple, dU).
BRANCH_ONLY=1 keeps only the positions that carry a branch pair (enough for train_branch.py; ~10x smaller).
Usage: PYTHONPATH=. python prep/go2pkl.py data/sp_x.jsonl TAG"""
import os
import sys, json, pickle, numpy as np
from lexmem.core import *
lex = Lexicon("data/NWL23.txt"); games = []; nbr = 0
def tup(d):   # Go move descriptor -> our move tuple (s, r, c, d, L, widx) or exchange (0, 0, 0, 2, 0, keep)
    return (d[6], d[1], d[2], d[3], d[4], d[5]) if d[0] == 0 else (0, 0, 0, 2, 0, tuple(int(x) for x in d[1:28]))
for line in open(sys.argv[1]):
    g = json.loads(line); traj = []
    for p in g["pos"]:
        board = np.full((N, N), -1, dtype=np.int16); blanks = np.zeros((N, N), dtype=bool)
        for i, ch in enumerate(p["b"]):
            if ch != ".": board[i // 15, i % 15] = IDX[ch.upper()]; blanks[i // 15, i % 15] = ch.islower()
        rack = np.array(p["rk"], dtype=np.int16); d = p["mv"]
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=np.array(p["un"], dtype=np.int16), scores=tuple(p["sc"]), bag=p["bag"], turn=p["t"], mover=p["p"])
        if d[0] == 0:
            rec["move"] = ("play", d[1], d[2], d[3], d[4], designate(lex, board, rack, d[1], d[2], d[3], d[4], d[5]), d[6])
        elif d[0] == 1: rec["move"] = ("exchange", tuple(int(x) for x in d[1:28]))
        else: rec["move"] = ("pass",)
        if "br" in p:
            alt = [int(x) for x in p["br"][1:]]; rec["branch"] = (tup(alt), float(p["br"][0])); nbr += 1
        traj.append(rec)
    if os.environ.get("BRANCH_ONLY") == "1": traj = [r for r in traj if "branch" in r]
    games.append((traj, tuple(g["final"])))
pickle.dump(games, open(f"data/games_{sys.argv[2]}.pkl", "wb"))
print(f"{len(games)} games, {sum(len(t) for t, _ in games)} positions, {nbr} branch pairs -> data/games_{sys.argv[2]}.pkl")
