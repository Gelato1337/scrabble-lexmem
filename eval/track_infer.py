"""Rack tracking over the whole move history (round 7): how much of the opponent's kept tiles does exact tracking over all of
its moves recover, against a static HastyBot (go/lexmem_track; go/lexmem_invert is the same for the last move alone)?
Positions: the rack-inference held-out positions (data/infer_test.npz, the fixed 3000-position sample of train_lexinfer.py).
History: the position's game in data/games_hasty_test.pkl (HastyBot v HastyBot; every turn with the mover's rack).
prep [N]
    -> data/trackpos_test.jsonl, one line per position: {"i", "leave": the true kept tiles, "steps": every move of the
    opponent from the start of the game to the position's play}; a step =
      cgp    the position before the move, the opponent on turn with its true rack / our rack at that time
      type   play (coords, tiles, played = the placed tiles) | exch (k = the number of tiles) | pass
      upre   the tiles unseen to us right after its previous move (first move: at the start of the game)
      draw   the tiles we drew between that moment and this move (our own turn in between)
    Lines are ordered by the size of the enumeration (racks per step before the history rules any out), largest first, so
    the longest chains start first; the sizes are printed.
check REF_A REF_B
    Two outputs of go/lexmem_track / go/lexmem_invert on the same positions: equal posteriors? (HIST=1 against lexmem_invert,
    FAST=0 against the default.)
analyze REF ...
    data/<REF>.jsonl -> log-loss of the true kept counts summed over the 27 tiles (bits, the measure of
    eval/infer_macondo.py) on the positions every REF answered, bits saved over the no-inference baseline, paired against
    the first REF, by the number of opponent moves in the history; tracking diagnostics.
okeep REF COND
    -> data/planinf_<COND>.jsonl (input of go/lexmem_simrack and go/lexmem_plan): data/planinf_uniform.jsonl with the
    posterior samples of data/<REF>.jsonl as "okeep"; positions without samples take data/planinf_base.jsonl's."""
import sys, json, pickle, random, re, types, numpy as np
from lexmem.core import LET, IDX, BLANK, FULL_BAG
from lexmem.harness import HastyBot
from lexmem.infer import baseline_logp
NEVP = 3000
TE = dict(np.load("data/infer_test.npz")); ev_idx = random.Random(7).sample(range(len(TE["gi"])), min(NEVP, len(TE["gi"])))   # = train_lexinfer.py
rs = lambda r27: "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(int(r27[t])))
FULL = FULL_BAG.astype(np.int64)
cgp = lambda *a, **k: HastyBot.cgp(types.SimpleNamespace(lexicon="NWL23"), *a, **k)

def on_board(rec):
    t = np.where(rec["blanks"], BLANK, rec["board"].astype(np.int64))[rec["board"] >= 0]
    return np.bincount(t, minlength=27).astype(np.int64)

def used(rec):
    """the tiles the mover took off its rack: placed (play) or put back into the bag (exchange)"""
    m = rec["move"]; v = np.zeros(27, dtype=np.int64)
    if m[0] == "play":
        for ch in m[5]:
            if ch != ".": v[BLANK if ch.islower() else IDX[ch]] += 1
    elif m[0] == "exchange":
        for ch in re.match(r"\(exch\s+([A-Za-z?]+)\)", m[1].strip()).group(1): v[BLANK if ch == "?" else IDX[ch.upper()]] += 1
    return v

def n_multisets(pool, n):
    c = [1] + [0] * n
    for p in pool:
        c = [sum(c[k - j] for j in range(min(int(p), k) + 1)) for k in range(n + 1)]
    return c[n]

def chain(games, i):
    """the opponent's moves up to position i's play, as steps for go/lexmem_track; (steps, None) or (None, reason)"""
    gi, ti = int(TE["gi"][i]), int(TE["turn"][i]); tr = games[gi][0]
    if any(tr[t]["mover"] != (tr[0]["mover"] + t) % 2 for t in range(ti + 2)): return None, "turn order"
    R = lambda t: tr[t]["rack"].astype(np.int64)
    U = lambda t: FULL - on_board(tr[t]) - R(t + 1)   # unseen to us before the opponent's move t (we hold our next turn's rack)
    steps = []
    for t in range(ti % 2, ti + 1, 2):   # the opponent's turns
        rec = tr[t]; m = rec["move"]
        if R(t).sum() != 7: return None, "short rack"
        if t < 2:   # its first move: our first rack, and our first draw if we moved first
            upre = FULL - (R(0) if t == 1 else R(1)); draw = R(2) - (R(0) - used(tr[0])) if t == 1 else np.zeros(27, dtype=np.int64)
        else:
            upre = U(t - 2) - (used(tr[t - 2]) if tr[t - 2]["move"][0] == "play" else 0); draw = R(t + 1) - (R(t - 1) - used(tr[t - 1]))
        back = used(tr[t - 1]) if t >= 1 and tr[t - 1]["move"][0] == "exchange" else 0   # what we put back into the bag
        if (draw < 0).any() or (upre < 0).any() or not np.array_equal(U(t), upre - draw + back): return None, "tile count"
        st = dict(cgp=cgp(rec["board"], rec["blanks"], R(t), tuple(int(x) for x in rec["scores"]), 0, rack2=R(t + 1)), upre=rs(upre), draw=rs(draw))
        if m[0] == "play":
            _, r0, c0, d0, L0, letters, s = m; col = "ABCDEFGHIJKLMNO"[c0]; P = used(rec)
            st.update(type="play", coords=f"{r0 + 1}{col}" if d0 == 0 else f"{col}{r0 + 1}", tiles=letters, played=rs(P)); st["_n"] = n_multisets(U(t) - P, 7 - int(P.sum()))
        elif m[0] == "exchange": st.update(type="exch", k=int(used(rec).sum())); st["_n"] = n_multisets(U(t), 7)
        else: st["type"] = "pass"; st["_n"] = n_multisets(U(t), 7)
        steps.append(st)
    if not np.array_equal(used(tr[ti]), TE["P"][i]) or not np.array_equal(R(ti) - used(tr[ti]), TE["leave"][i]): return None, "position"
    return steps, None

def prep(npos):
    games = pickle.load(open("data/games_hasty_test.pkl", "rb")); bad = {}; ns = {"play": [], "exch": [], "pass": []}; first = []; nst = []; lines = []
    for i in ev_idx[:npos]:
        steps, why = chain(games, i)
        if steps is None: bad[why] = bad.get(why, 0) + 1; continue
        cost = 0
        for k, st in enumerate(steps):
            n = st.pop("_n"); ns[st["type"]].append(n); cost += n
            if k == 0: first.append(n)
        nst.append(len(steps)); lines.append((cost, json.dumps(dict(i=int(i), leave=rs(TE["leave"][i]), steps=steps))))
    with open("data/trackpos_test.jsonl", "w") as f:
        for _, line in sorted(lines, key=lambda x: -x[0]): f.write(line + "\n")
    nst = np.array(nst)
    print(f"{len(nst)} positions -> data/trackpos_test.jsonl; left out {bad}; opponent moves per position mean {nst.mean():.2f}, "
          f"median {np.median(nst):.0f}, max {nst.max()}; one move only: {(nst == 1).sum()}", flush=True)
    for k, v in ns.items():
        if v: v = np.array(v, dtype=np.float64); print(f"  {k}: {len(v)} steps, racks per step mean {v.mean():.0f}, median {np.median(v):.0f}, max {v.max():.0f}, total {v.sum():.3g}")
    print(f"  first steps (nothing ruled out by the history): {np.sum(first):.3g} racks; all steps {sum(np.sum(v) for v in ns.values()):.3g}", flush=True)

def load(ref):
    return {o["i"]: o for o in map(json.loads, open(f"data/{ref}.jsonl"))}

def check(a, b):
    A, B = load(a), load(b); ids = sorted(set(A) & set(B)); both = [i for i in ids if A[i].get("complete") and B[i].get("complete")]
    dm = [float(np.abs(np.array(A[i]["marg"]) - np.array(B[i]["marg"])).max()) for i in both]
    print(f"{a} vs {b}: {len(ids)} common positions, {len(both)} answered by both ({sum(bool(A[i].get('complete')) for i in ids)} / {sum(bool(B[i].get('complete')) for i in ids)})")
    if not both: return
    print(f"  consistent kept sets differ in {sum(A[i]['nmatch'] != B[i]['nmatch'] for i in both)}; largest difference of a marginal {max(dm):.2e} "
          f"(> 1e-9 in {sum(d > 1e-9 for d in dm)}); posterior of the true kept set differs (> 1e-9) in {sum(abs(A[i]['post'] - B[i]['post']) > 1e-9 for i in both)}; "
          f"samples differ in {sum(A[i].get('samples') != B[i].get('samples') for i in both if A[i].get('samples') and B[i].get('samples'))}")
    for k in ("cands", "slow", "secs"):
        if all(k in o for o in A.values()): print(f"  {a} {k}: total {sum(o[k] for o in A.values()):.4g}", end="")
    print()

def bits(o, i):
    Y = np.minimum(TE["leave"][i], 3); M = np.clip(np.array(o["marg"]), 1e-9, 1.0)
    return float(-np.log2(M[np.arange(27), Y]).sum())

def analyze(refs):
    R = {r: {i: o for i, o in load(r).items() if o.get("complete")} for r in refs}   # complete: a posterior (or the prior when none is consistent)
    ids = sorted(set.intersection(*[set(v) for v in R.values()]))
    base = np.array([-baseline_logp(TE["U"][i], TE["P"][i], int(TE["leave"][i].sum()))[np.arange(27), np.minimum(TE["leave"][i], 3)].sum() / np.log(2) for i in ids])
    B = {r: np.array([bits(R[r][i], i) for i in ids]) for r in refs}; se = lambda v: v.std() / np.sqrt(len(v))
    nst = None
    print(f"{len(ids)} positions answered by every source; no-inference baseline {base.mean():.3f} bits (the true kept counts, summed over the 27 tiles)")
    for r in refs:
        sv = base - B[r]; line = f"  {r:22s} {B[r].mean():.3f} bits, saved {sv.mean():+.3f} ± {se(sv):.3f} (median {np.median(sv):+.3f})"
        lift = np.array([R[r][i]["lift"] for i in ids]); ok = lift > -900
        line += f"; joint lift of the true kept set {lift[ok].mean():+.2f} bits on {ok.sum()}"
        if r != refs[0]: d = B[r] - B[refs[0]]; line += f"; vs {refs[0]} {d.mean():+.3f} ± {se(d):.3f}"
        print(line)
        o = [R[r][i] for i in ids]
        if "steps" in o[0]:
            nst = np.array([x["steps"] for x in o])
            print(f"    opponent moves used mean {nst.mean():.2f}; the true rack lost its weight in {sum(x['lost'] >= 0 for x in o)} positions, steps skipped for size "
                  f"{sum(x['skipped'] for x in o)}, restarted without history {sum(x['empty'] for x in o)}; consistent kept sets {np.mean([x['nmatch'] for x in o]):.0f} of "
                  f"{np.mean([x['n'] for x in o]):.0f}; racks put to HastyBot {sum(x['cands'] for x in o):.3g} ({sum(x['slow'] for x in o):.3g} on the full move list), "
                  f"{sum(x['secs'] for x in o) / 3600:.1f} h summed over positions")
    full = [r for r in refs if "steps" in next(iter(R[r].values()))]
    if full and len(refs) > 1:   # by the length of the history, taken from the last source (give the whole history last)
        nst = np.array([R[full[-1]][i]["steps"] for i in ids]); print(f"bits saved by the number of opponent moves in the history ({full[-1]}):")
        for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 6), (7, 9), (10, 99)):
            k = (nst >= lo) & (nst <= hi)
            if k.sum(): print(f"  {lo}-{hi} moves ({k.sum()} positions): " + ", ".join(f"{r} {(base - B[r])[k].mean():+.3f}" for r in refs))

def okeep(ref, cond):
    S = {i: o.get("samples") for i, o in load(ref).items()}; base = {o["id"]: o["okeep"] for o in map(json.loads, open("data/planinf_base.jsonl"))}; miss = 0
    with open(f"data/planinf_{cond}.jsonl", "w") as f:
        for o in map(json.loads, open("data/planinf_uniform.jsonl")):
            k = S.get(o["id"])
            if not k: k = base[o["id"]]; miss += 1
            f.write(json.dumps(dict(id=o["id"], cgp=o["cgp"], okeep=k)) + "\n")
    print(f"data/{ref}.jsonl -> data/planinf_{cond}.jsonl; {miss} positions without samples use the baseline's", flush=True)

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "prep": prep(int(sys.argv[2]) if len(sys.argv) > 2 else NEVP)
    elif c == "check": check(sys.argv[2], sys.argv[3])
    elif c == "okeep": okeep(sys.argv[2], sys.argv[3])
    else: analyze(sys.argv[2:])
