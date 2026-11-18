"""Paired comparison of Go-harness game runs (cmd/lexmem_search: data/search_<tag>.jsonl, one line per game g with its spread;
game g has seed SEED + g in every run, our side first when g is even). Win = spread > 0, as in play_eval.py.
For each tag: win rate ± se, spread, moves by net/search vs Macondo; vs the baseline tag (first argument): mean per-game
difference of wins ± its standard error over the games both runs played.
Usage: python tools/search_games.py BASE_TAG TAG [TAG ...]"""
import sys, json, numpy as np
def load(tag):
    out = {}
    for ln in open(f"data/search_{tag}.jsonl"):
        r = json.loads(ln); out[r["g"]] = r
    return out
runs = {t: load(t) for t in sys.argv[1:]}
base = sys.argv[1]
for t, r in runs.items():
    s = np.array([v["spread"] for v in r.values()]); w = (s > 0).astype(float); n = np.array([v["n"] for v in r.values()]).sum(0)
    line = f"{t:>24}: {len(s)} games, win {100 * w.mean():.1f} % ± {100 * w.std() / np.sqrt(len(w)):.1f}, spread {s.mean():+.1f}, our moves net/search {n[0]} Macondo {n[1]} pass {n[2]}"
    if t != base:
        g = sorted(set(r) & set(runs[base]))
        d = np.array([(r[k]["spread"] > 0) - (runs[base][k]["spread"] > 0) for k in g], dtype=float)
        line += f" | vs {base} on {len(g)} paired games: {100 * d.mean():+.1f} ± {100 * d.std() / np.sqrt(len(d)):.1f}"
    print(line)
