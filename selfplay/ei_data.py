"""Expert-iteration data from Go self-play (round 5, stage 2): the search's estimates as soft policy targets.
positions TAG N PER_GAME
    data/search_<TAG>.jsonl (go/lexmem_search OURS=net OPP=net TRAJ=1: whole games, every turn recorded) ->
    data/searchpos_<TAG>.jsonl: up to PER_GAME positions per game (seed 0) where the mover played tiles, the bag holds >= 8 tiles
    (the search phase) and the board is not empty; shuffled; the first N. Fields i, cgp, turn (read by lexmem_search
    POSITIONS=...), g, ti.
merge TAG OUT LABEL_TAG [LABEL_TAG ...]
    The games of data/search_<TAG>.jsonl as training games data/games_<OUT>.pkl in the format of prep/label_bot.py (one
    (trajectory, final points by player) per game; records as lexmem/replay.py), with the search's labels from
    data/search_<LABEL_TAG>.jsonl (POSITIONS runs over data/searchpos_<TAG>.jsonl): a labelled position gets cands
    [(r, c, d, L, widx)] = the search's candidates and cand_est = their mean playout win % over the S worlds, known=True; every
    other position known=False (no policy target; it stays in the trajectory for the k-ply value targets). train/imitate.py EI=1
    then trains on softmax(cand_est / TAU) over the candidates.
values TAG OUT LABEL_TAG [LABEL_TAG ...]
    Value targets for our own search leaf (train/value_leaf.py): every labelled position of data/searchpos_<TAG>.jsonl as
    (record, y) in data/vleaf_<OUT>.pkl, y = the search's value of the position for the mover = mean playout win probability
    (over the S worlds) of the candidate the search plays (lexmem_search's Z = 1 rule: the policy's top move unless another
    candidate's mean paired gain exceeds one standard error). Also stored: y_top (policy's top move), y_max (best mean), g.
check OUT [USE_LEX]
    Loads data/games_<OUT>.pkl through train/imitate.py's MoveDataset (EI=1, TAU=2) and prints soft-target statistics."""
import os, sys, json, random, pickle, numpy as np
from lexmem.core import LET, IDX, N
sys.path.insert(0, "eval"); from search_positions import cgp, pick

def games_of(tag):
    return [json.loads(l) for l in open(f"data/search_{tag}.jsonl")]

def record(t):
    board = np.full((N, N), -1, dtype=np.int8); blanks = np.zeros((N, N), dtype=bool)
    for k, ch in enumerate(t["b"]):
        if ch != ".": board[k // N, k % N] = IDX[ch.upper()]; blanks[k // N, k % N] = ch.islower()
    if t["k"] == "play": mv = ("play", t["r"], t["c"], t["d"], len(t["w"]), t["w"], t["s"])
    else: mv = ("exchange" if t["k"] == "exch" else "pass",)
    return dict(board=board, blanks=blanks, rack=np.array(t["rk"], dtype=np.int16), unseen=np.array(t["un"], dtype=np.int16),
                scores=tuple(t["sc"]), bag=t["bag"], turn=t["t"], mover=t["p"], move=mv, known=False)

def positions(tag, n, per_game):
    rng = random.Random(0); pool = []
    for gr in games_of(tag):
        ok = [ti for ti, t in enumerate(gr["traj"]) if t["k"] == "play" and t["bag"] >= 8 and t["b"].strip(".")]
        pool += [(gr["g"], ti) for ti in rng.sample(ok, min(per_game, len(ok)))]
    rng.shuffle(pool); games = {gr["g"]: gr for gr in games_of(tag)}
    with open(f"data/searchpos_{tag}.jsonl", "w") as f:
        for i, (g, ti) in enumerate(pool[:n]):
            t = games[g]["traj"][ti]; rec = record(t)
            f.write(json.dumps(dict(i=i, cgp=cgp(rec["board"], rec["blanks"], rec["rack"], rec["scores"], t["z"]), turn=t["t"], g=g, ti=ti)) + "\n")
    print(f"{min(n, len(pool))} positions (of {len(pool)}) from {len(games)} games -> data/searchpos_{tag}.jsonl")

def merge(tag, out, label_tags):
    where = {p["i"]: (p["g"], p["ti"]) for p in map(json.loads, open(f"data/searchpos_{tag}.jsonl"))}
    lab = {}
    for lt in label_tags:
        for o in map(json.loads, open(f"data/search_{lt}.jsonl")):
            if o["v"]:   # one candidate only: nothing to choose
                cs = [(c[1], c[2], c[3], c[4], c[5]) for c in o["c"] if c[0] == 0]; es = [100 * float(np.mean(v)) for c, v in zip(o["c"], o["v"]) if c[0] == 0]
                lab[where[o["i"]]] = (cs, es)
    games, nl = [], 0
    for gr in games_of(tag):
        traj = [record(t) for t in gr["traj"]]
        for ti, rec in enumerate(traj):
            if (gr["g"], ti) in lab: rec["cands"], rec["cand_est"] = lab[(gr["g"], ti)]; rec["known"] = True; nl += 1
        games.append((traj, tuple(gr["fp"])))
    pickle.dump(games, open(f"data/games_{out}.pkl", "wb"))
    print(f"{len(games)} games, {sum(len(t) for t, _ in games)} positions, {nl} labelled ({len(lab)} labels) -> data/games_{out}.pkl")

def values(tag, out, label_tags):
    where = {p["i"]: (p["g"], p["ti"]) for p in map(json.loads, open(f"data/searchpos_{tag}.jsonl"))}
    games = {gr["g"]: gr for gr in games_of(tag)}; res = []
    for lt in label_tags:
        for o in map(json.loads, open(f"data/search_{lt}.jsonl")):
            if not o["v"]: continue   # one candidate: no search value
            V = np.array(o["v"], dtype=np.float64); m = V.mean(1); g, ti = where[o["i"]]
            res.append(dict(rec=record(games[g]["traj"][ti]), y=float(m[pick(V, V.shape[1], 1.0)]), y_top=float(m[0]), y_max=float(m.max()), g=g))
    tmp = f"data/vleaf_{out}.pkl.{os.getpid()}"; pickle.dump(res, open(tmp, "wb")); os.replace(tmp, f"data/vleaf_{out}.pkl")   # atomic: parallel jobs build it
    y = np.array([r["y"] for r in res])
    print(f"{len(res)} positions -> data/vleaf_{out}.pkl: y mean {100 * y.mean():.1f} %, sd {100 * y.std():.1f}; "
          f"y - y_top {100 * np.mean([r['y'] - r['y_top'] for r in res]):+.2f}, y_max - y {100 * np.mean([r['y_max'] - r['y'] for r in res]):+.2f} win-% points")

def check(out, use_lex):
    from lexmem.core import Lexicon
    from lexmem.policy import MoveDataset
    games = pickle.load(open(f"data/games_{out}.pkl", "rb")); ds = MoveDataset(Lexicon("data/NWL23.txt"), games, seed=0, use_lex=use_lex, tau=2.0, ei=True)
    ds.rng = random.Random(0); its = [ds[i] for i in range(200)]
    top = np.mean([int(np.argmax(it["soft"]) == it["target"]) for it in its]); mg = np.array([float(it["margin"]) for it in its])
    print(f"{len(ds.items)} training positions; 200 sampled: soft target's top = the move played {100 * top:.1f} %, mean top-2 margin {mg.mean():.2f} "
          f"win-% points, margin >= 3 in {100 * (mg >= 3).mean():.1f} %, soft max {np.mean([it['soft'].max() for it in its]):.2f}, moves per position "
          f"{np.mean([len(it['widx']) for it in its]):.0f}" + (f", retrieval words {np.mean([len(it['rmask'][0]) for it in its]):.0f}" if use_lex else ""))

if __name__ == "__main__":
    if sys.argv[1] == "positions": positions(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "merge": merge(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "values": values(sys.argv[2], sys.argv[3], sys.argv[4:])
    else: check(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "1")
