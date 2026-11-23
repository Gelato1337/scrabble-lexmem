"""Macondo's rack inference as the reference for the rack-inference benchmark (round 6).
prep N
    The first N held-out positions of train/train_lexinfer.py's evaluation set (data/infer_test.npz, same sample) as input
    lines for go/lexmem_infer -> data/inferpos_test.jsonl (position before the opponent's play with the opponent on turn and
    rack = the tiles it placed / our rack, the play, the true leave).
analyze [MODEL_TAG ...]
    data/<REF>.jsonl (go/lexmem_infer or go/lexmem_invert output; env REF, default inferref_test) -> Macondo's log-loss of the true kept counts summed over the 27 tiles
    (bits; the same factorised measure as the models), the no-inference baseline, and for each model tag its per-position
    bits (data/<TAG>_evbits.npy, same positions) paired on the positions Macondo answered."""
import os, sys, json, random, types, numpy as np
from lexmem.core import LET, BLANK, N, FULL_BAG
from lexmem.harness import HastyBot
from lexmem.infer import baseline_logp
NEVP = 3000
TE = dict(np.load("data/infer_test.npz")); ev_idx = random.Random(7).sample(range(len(TE["gi"])), min(NEVP, len(TE["gi"])))   # = train_lexinfer.py (its NEV_POS prefix)
rs = lambda r27: "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(int(r27[t])))
cgp = lambda *a, **k: HastyBot.cgp(types.SimpleNamespace(lexicon="NWL23"), *a, **k)

def prep(n):
    with open("data/inferpos_test.jsonl", "w") as f:
        for i in ev_idx[:n]:
            board, blanks = TE["board"][i].astype(np.int64), TE["blanks"][i].astype(bool); P, U = TE["P"][i], TE["U"][i]
            t = np.where(blanks, BLANK, board)[board >= 0]; mine = FULL_BAG - np.bincount(t, minlength=27) - U   # the observer's rack
            r0, c0, d0, L0, w0, s = (int(x) for x in TE["move"][i]); col = "ABCDEFGHIJKLMNO"[c0]
            f.write(json.dumps(dict(i=int(i), cgp=cgp(board, blanks, P, tuple(int(x) for x in TE["scores"][i]), 0, rack2=mine),
                                    coords=f"{r0 + 1}{col}" if d0 == 0 else f"{col}{r0 + 1}", tiles=str(TE["letters"][i]), played=rs(P), leave=rs(TE["leave"][i]))) + "\n")
    print(f"{n} positions -> data/inferpos_test.jsonl")

def analyze(tags):
    pos = {i: k for k, i in enumerate(ev_idx)}; rows = {}
    errs = 0
    ref = os.environ.get("REF", "inferref_test")
    for o in map(json.loads, open(f"data/{ref}.jsonl")):
        if o.get("err") and (not o.get("n") or o["err"].startswith("too many")): errs += 1; continue
        i = o["i"]; Y = np.minimum(TE["leave"][i], 3); M = np.clip(np.array(o["marg"]), 1e-9, 1.0)
        mb = -np.log2(M[np.arange(27), Y]).sum(); bb = -baseline_logp(TE["U"][i], TE["P"][i], int(TE["leave"][i].sum()))[np.arange(27), Y].sum() / np.log(2)
        rows[i] = (mb, bb, int(TE["bag"][i]), o["secs"], o["lift"])
    ks = sorted(rows, key=lambda i: pos[i]); R = np.array([rows[i][:4] for i in ks])
    print(f"Macondo rangefinder ({ref}) on {len(ks)} positions ({errs} without an answer): {R[:,0].mean():.3f} bits, baseline {R[:,1].mean():.3f}, "
          f"saved {(R[:,1]-R[:,0]).mean():+.3f} ± {(R[:,1]-R[:,0]).std()/np.sqrt(len(R)):.3f}; {R[:,3].mean():.1f} s per position")
    sv = R[:, 1] - R[:, 0]; print(f"  median saved {np.median(sv):+.3f}; positions > 5 bits worse than the baseline: {(sv < -5).sum()}")
    tm = [o for o in map(json.loads, open(f"data/{ref}.jsonl")) if "truematch" in o]
    if tm: print(f"  exact inversion: the true kept set reproduces the play in {sum(o['truematch'] for o in tm)} of {len(tm)} positions; "
                 f"consistent kept sets {np.mean([o['nmatch'] for o in tm]):.0f} of {np.mean([o['n'] for o in tm]):.0f} on average"
                 + (f"; the play is one of several top-equity ties in {sum(o['truematch'] and o['tietrue'] > 1 for o in tm)}" if all("tietrue" in o for o in tm) else ""))
    lifts = np.array([rows[i][4] for i in ks]); ok = lifts > -900
    print(f"  Macondo's own lift for the true leave (joint, log2 posterior/prior): mean {lifts[ok].mean():+.2f} bits on {ok.sum()} positions")
    for tag in tags:
        E = np.load(f"data/{tag}_evbits.npy"); m = np.array([E[pos[i], 0] for i in ks]); d = m - R[:, 0]
        print(f"  {tag}: {m.mean():.3f} bits on the same positions, saved {(R[:,1]-m).mean():+.3f}; model - Macondo {d.mean():+.3f} ± {d.std()/np.sqrt(len(d)):.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(int(sys.argv[2]))
    else: analyze(sys.argv[2:])
