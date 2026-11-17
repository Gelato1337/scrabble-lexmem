"""Paired comparison of planner game runs (play_eval.py pickles: a[g] = spread of game g, seed SEED0 + g, same for every player).
Win rates (spread > 0, as in play_eval) and lex - no-lex differences with the standard error of the per-game paired difference,
per training size and seed and pooled over seeds; NWL and CSW.
Usage: python tools/paired_games.py [NG=4000]"""
import sys, os, pickle, numpy as np
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
def load(plan, ck, dic):
    f = f"data/eval_policy_imit_nolex_{plan}-{ck}{'_CSW24' if dic == 'csw' else ''}_{NG}.pkl"
    return (pickle.load(open(f, "rb")) > 0).astype(float) if os.path.exists(f) else None
for dic in ("nwl", "csw"):
    print(f"== {dic.upper()}")
    for p in ("topscore", "exact"):
        w = load(p, "", dic)
        if w is not None: print(f"  {p:>8}: {100 * w.mean():.1f} % ± {100 * w.std() / np.sqrt(len(w)):.1f}")
    for n in ("3000", "10000", "all"):
        d_all = []
        for s in (0, 1):
            a, b = load("net", f"lexplan_nolex_{n}_s{s}", dic), load("net", f"lexplan_lex_{n}_s{s}", dic)
            if a is None or b is None: continue
            d = b - a; d_all.append(d)
            print(f"  {n:>5} s{s}: no-lex {100 * a.mean():.1f}  lex {100 * b.mean():.1f}  lex - no-lex {100 * d.mean():+.1f} ± {100 * d.std() / np.sqrt(len(d)):.1f}"
                  f"  (outcome agreement {np.mean(a == b):.2f})")
        if len(d_all) == 2:
            d = np.concatenate(d_all); print(f"  {n:>5} pooled: lex - no-lex {100 * d.mean():+.1f} ± {100 * d.std() / np.sqrt(len(d)):.1f}")
