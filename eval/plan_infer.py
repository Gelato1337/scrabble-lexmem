"""Planning with rack inference (round 6): does knowing what the opponent kept change the planning decision, and how much of
that does each inference source recover?
Positions: the rack-inference held-out positions (data/infer_test.npz, the fixed 3000-position sample of train_lexinfer.py),
from the side of the player who moves next (the observer): board after the opponent's (HastyBot's) play, our rack, scores.
Decision: the one-turn exact planner of the planning benchmark (go/lexmem_plan, top K = 16 plays by score, S = 256 shared
samples, PV = score - E_opp + E_my). Conditions differ only in the opponent's rack in each sample:
  uniform   7 tiles from all tiles unseen to us (the benchmark's labels; no inference)
  base      kept tiles drawn from the exact no-inference posterior (it held the placed tiles P) + refill
  <TAG>     kept tiles drawn from a rack-inference net's prediction (data/<TAG>_evlogp.npy): the joint closest to the baseline
            with the net's per-tile marginals (lexmem.infer.fit_tilt) + refill
  exact     kept tiles drawn from the exact inversion of a static HastyBot (go/lexmem_invert SAMPLES=256) + refill
  true      the true kept tiles + refill (perfect inference); true_eval = the same with independent samples (SEED0)
Regret of each condition's choice is measured under true_eval (PV with the true kept tiles, independent samples), so the
"true" condition's regret is the sampling-noise floor.
prep [N]            -> data/planinf_<cond>.jsonl (lexmem_plan input with "okeep"); env TAGS (net tags), EXACT (lexmem_invert
                       output with samples, default data/inferinv_test_s.jsonl), NPROC
analyze [cond ...]  <- data/planinf_out_<cond>.jsonl"""
import os, sys, json, random, types, numpy as np
from multiprocessing import Pool
from lexmem.core import LET, BLANK, FULL_BAG
from lexmem.harness import HastyBot
from lexmem.replay import place
from lexmem.infer import fit_tilt, sample_leaves
S = 256
TE = dict(np.load("data/infer_test.npz")); ev_idx = random.Random(7).sample(range(len(TE["gi"])), min(3000, len(TE["gi"])))   # = train_lexinfer.py
rs = lambda r27: "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(int(r27[t])))
cgp = lambda *a, **k: HastyBot.cgp(types.SimpleNamespace(lexicon="NWL23"), *a, **k)

def mover_cgp(i):
    board, blanks = TE["board"][i].astype(np.int64), TE["blanks"][i].astype(bool); P, U = TE["P"][i], TE["U"][i]
    t = np.where(blanks, BLANK, board)[board >= 0]; mine = FULL_BAG - np.bincount(t, minlength=27) - U   # the observer's rack
    r0, c0, d0, L0, w0, s = (int(x) for x in TE["move"][i]); b2, bl2, _ = place(board, blanks, r0, c0, d0, str(TE["letters"][i]))
    opp, obs = (int(x) for x in TE["scores"][i])
    return cgp(b2, bl2, mine, (obs, opp + s), 0)   # our rack, no opponent rack: the CGP bag = every tile unseen to us

def net_samples(args):
    i, k, logp = args
    U, P, n = TE["U"][i].astype(int), TE["P"][i].astype(int), int(TE["leave"][i].sum())
    phi, err = fit_tilt(U, P, n, logp)
    return [rs(l) for l in sample_leaves(U, P, n, phi, S, np.random.default_rng(i))], err

def prep(npos):
    idx = ev_idx[:npos]; lines = {i: mover_cgp(i) for i in idx}
    def write(cond, keeps):
        with open(f"data/planinf_{cond}.jsonl", "w") as f:
            for i in idx:
                o = dict(id=int(i), cgp=lines[i])
                if keeps is not None: o["okeep"] = keeps[i]
                f.write(json.dumps(o) + "\n")
        print(f"{len(idx)} positions -> data/planinf_{cond}.jsonl", flush=True)
    write("uniform", None)
    write("true", {i: [rs(TE["leave"][i])] for i in idx})
    write("base", {i: [rs(l) for l in sample_leaves(TE["U"][i].astype(int), TE["P"][i].astype(int), int(TE["leave"][i].sum()), np.zeros((27, 4)), S,
                                                    np.random.default_rng(i))] for i in idx})
    ex = os.environ.get("EXACT", "data/inferinv_test_s.jsonl")
    if os.path.exists(ex):
        E = {o["i"]: o.get("samples") for o in map(json.loads, open(ex))}
        miss = [i for i in idx if not E.get(i)]
        write("exact", {i: E[i] if E.get(i) else [rs(l) for l in sample_leaves(TE["U"][i].astype(int), TE["P"][i].astype(int), int(TE["leave"][i].sum()),
                                                                              np.zeros((27, 4)), S, np.random.default_rng(i))] for i in idx})
        print(f"  exact: {len(miss)} positions without inversion samples use the baseline", flush=True)
    with Pool(int(os.environ.get("NPROC", 32))) as pool:
        for tag in os.environ.get("TAGS", "").split():
            LP = np.load(f"data/{tag}_evlogp.npy")
            res = pool.map(net_samples, [(i, k, LP[k]) for k, i in enumerate(idx)], chunksize=8)
            errs = np.array([e for _, e in res])
            print(f"  {tag}: marginal fit error mean {errs.mean():.4f}, max {errs.max():.4f}", flush=True)
            write(tag, {i: r[0] for i, r in zip(idx, res)})

def load(cond):
    out = {}
    for o in map(json.loads, open(f"data/planinf_out_{cond}.jsonl")):
        if o.get("err") or not o.get("cands"): continue
        out[o["id"]] = [(tuple(c["d"]), c["d"][6] - c["eo"] + c["em"]) for c in o["cands"]]
    return out

def analyze(conds):
    T = load("true_eval"); C = {c: load(c) for c in conds}
    ids = sorted(set(T).intersection(*[set(v) for v in C.values()]))
    keyok = [i for i in ids if all([d for d, _ in C[c][i]] == [d for d, _ in T[i]] for c in conds)]
    print(f"{len(ids)} positions labelled in every condition, {len(keyok)} with identical candidate lists")
    reg = {}
    for c in conds + ["score"]:
        r = []
        for i in keyok:
            pv = np.array([v for _, v in T[i]])
            ch = 0 if c == "score" else int(np.argmax([v for _, v in C[c][i]]))   # candidates are sorted by score
            r.append(pv.max() - pv[ch])
        reg[c] = np.array(r)
    rel = np.array([int(np.argmax([v for _, v in C["true"][i]])) != int(np.argmax([v for _, v in C["uniform"][i]])) for i in keyok]) \
        if "true" in conds and "uniform" in conds else np.ones(len(keyok), bool)
    print(f"regret in points under the true kept tiles (independent samples); 'changed' = the {rel.sum()} positions where perfect "
          f"inference changes the uniform planner's choice")
    for c in ["score"] + conds:
        r = reg[c]; print(f"  {c:28s} all {r.mean():.3f} ± {r.std()/np.sqrt(len(r)):.3f} | changed {r[rel].mean():.3f}")
    base = "uniform" if "uniform" in conds else conds[0]
    print(f"paired vs {base} (negative = better):")
    for c in conds:
        if c == base: continue
        d = reg[c] - reg[base]; print(f"  {c:28s} all {d.mean():+.3f} ± {d.std()/np.sqrt(len(d)):.3f} | changed {d[rel].mean():+.3f} ± {d[rel].std()/np.sqrt(max(rel.sum(),1)):.3f}")
    lex = [c for c in conds if "_lex_" in c]; nol = [c for c in conds if "_nolex_" in c]
    if lex and nol:
        d = np.mean([reg[c] for c in lex], 0) - np.mean([reg[c] for c in nol], 0)
        print(f"memory - control (seed means): all {d.mean():+.3f} ± {d.std()/np.sqrt(len(d)):.3f} | changed {d[rel].mean():+.3f} ± {d[rel].std()/np.sqrt(max(rel.sum(),1)):.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    else: analyze(sys.argv[2:])
