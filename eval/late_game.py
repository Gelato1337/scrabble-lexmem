"""Exact solving earlier, late-game playouts (CPU only; lumi/late_game.sh, go/lexmem_search START).
Does SIMMING_BOT win more when Macondo's pre-endgame solver takes over with 2 or 3 tiles in the bag instead of 1?
Positions: from the HastyBot v HastyBot held-out games (data/games_hasty_test.pkl), the first turn of a game at which exactly
BAG tiles are in the bag and both players hold 7 tiles; the mover is our side, both racks as they were.
Each position is played to the end of the game against HastyBot by each variant of our bot (the same bag seed per position).
prep BAG [N]             -> data/late_bag<BAG>.jsonl ({"i": game index, "cgp": both racks, the mover on turn})
analyze TAG_A TAG_B ...  data/search_<TAG>.jsonl (one line per position: "g", "spread"): win-% of each run and, paired against
                         the first, win-% difference ± se (a tie counts half), share of positions with another result, spread."""
import sys, json, math

def prep(bag, n):
    import pickle, types, numpy as np
    from lexmem.core import BLANK, FULL_BAG
    from lexmem.harness import HastyBot
    cgp = lambda *a, **k: HastyBot.cgp(types.SimpleNamespace(lexicon="NWL23"), *a, **k)
    games = pickle.load(open("data/games_hasty_test.pkl", "rb")); k = 0; sp = []
    with open(f"data/late_bag{bag}.jsonl", "w") as f:
        for gi, (tr, _) in enumerate(games):
            for t in range(len(tr) - 1):
                rec, nxt = tr[t], tr[t + 1]
                if rec["bag"] != bag or nxt["mover"] == rec["mover"] or rec["rack"].sum() != 7 or nxt["rack"].sum() != 7: continue
                tl = np.where(rec["blanks"], BLANK, rec["board"].astype(np.int64))[rec["board"] >= 0]
                left = FULL_BAG.astype(np.int64) - np.bincount(tl, minlength=27) - rec["rack"] - nxt["rack"]
                if (left < 0).any() or left.sum() != bag: continue
                f.write(json.dumps(dict(i=gi, cgp=cgp(rec["board"], rec["blanks"], rec["rack"].astype(np.int64), tuple(int(x) for x in rec["scores"]), 0,
                                                      rack2=nxt["rack"].astype(np.int64)))) + "\n")
                k += 1; sp.append(int(rec["scores"][0]) - int(rec["scores"][1])); break
            if k >= n: break
    sp = np.array(sp)
    print(f"{k} positions with {bag} tiles in the bag -> data/late_bag{bag}.jsonl (of {len(games)} games); mover's spread mean {sp.mean():+.1f}, "
          f"within 30 points in {(abs(sp) <= 30).mean() * 100:.0f} %", flush=True)

def ms(v):
    n = len(v); m = sum(v) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n) if n > 1 else float("nan"))

def analyze(tags):
    R = {t: {o["g"]: o["spread"] for o in map(json.loads, open("data/search_%s.jsonl" % t))} for t in tags}
    ids = sorted(set.intersection(*[set(v) for v in R.values()])); w = lambda s: 100.0 if s > 0 else 50.0 if s == 0 else 0.0
    print("%d positions played out by every run" % len(ids))
    for t in tags:
        wv = [w(R[t][i]) for i in ids]; sv = [R[t][i] for i in ids]
        line = "  %-22s win-%% %.2f spread %+.2f" % (t, ms(wv)[0], ms(sv)[0])
        if t != tags[0]:
            b = tags[0]; dw = ms([w(R[t][i]) - w(R[b][i]) for i in ids]); ds = ms([R[t][i] - R[b][i] for i in ids])
            line += " | vs %s: win-%% %+.2f +- %.2f, spread %+.2f +- %.2f, another winner in %.1f %%, another spread in %.1f %%" % (
                b, dw[0], dw[1], ds[0], ds[1], 100 * sum(w(R[t][i]) != w(R[b][i]) for i in ids) / len(ids), 100 * sum(R[t][i] != R[b][i] for i in ids) / len(ids))
        print(line)

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    else: analyze(sys.argv[2:])
