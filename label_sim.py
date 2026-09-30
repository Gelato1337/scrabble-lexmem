"""Label positions with 2-ply Monte-Carlo sim estimates for their top-K candidates (CPU, many procs).
Output: data/simlabels_{TAG}.pkl = list of dicts: position fields + cands [(score,r,c,d,L,widx,letters,equity)] + est [K floats].
Usage: NPROC=48 TAG=hasty300k python label_sim.py <n_positions> [seed]"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
E = os.environ.get; NPOS = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NPROC = int(E("NPROC", 8)); TAG = E("TAG", "lab"); K = int(E("K", 8)); S = int(E("S", 64)); LEXICON = E("LEXICON", "data/NWL23.txt")
GAMES = E("GAMES", "data/games_hasty_train.pkl"); ROLLOUT = E("ROLLOUT", "hasty"); FUT = E("FUT", "0") == "1"
G = {}
def init():
    from lexmem.core import Lexicon
    from lexmem.macondo_gen import MacondoGen
    G["lex"] = Lexicon(LEXICON); G["mg"] = MacondoGen()
def one(task):
    try: return _one(task)
    except Exception as e:
        print(f"skip position {task[0]}: {type(e).__name__}: {e}", flush=True); return None
def _one(task):
    """task = (i, rec): the parent samples positions and ships only the needed fields (workers never load the games file)."""
    from lexmem.sim import sim_eval
    i, rec = task; lex, mg = G["lex"], G["mg"]; rng = random.Random(SEED0 + i)
    board, blanks, rack, unseen = rec["board"], rec["blanks"], rec["rack"], rec["unseen"]
    moves = mg.gen(lex, board, blanks, rack, rec["scores"], 0, n=400)
    if len(moves) < 2: return None
    cands = moves[:K]
    pool = [t for t in range(27) for _ in range(int(unseen[t]))]; worlds = []
    for _ in range(S): rng.shuffle(pool); worlds.append(list(pool))
    record = [] if FUT else None
    sums = sim_eval(None, None, lex, mg, board, blanks, rack, unseen, rec["scores"], rec["turn"], cands, worlds, ROLLOUT, record=record)
    fut = None
    if FUT and record:   # aggregate identical (cand, side, lane, word) -> count
        arr = np.array(record, dtype=np.int32); u, cnt = np.unique(arr, axis=0, return_counts=True)
        fut = np.concatenate([u, cnt[:, None].astype(np.int32)], 1)
    return dict(board=board, blanks=blanks, rack=rack, unseen=unseen, scores=rec["scores"], bag=rec["bag"], turn=rec["turn"],
                cands=[tuple(m) for m in cands], est=(sums / S).tolist(), game=rec["game"], fut=fut, S=S)
if __name__ == "__main__":
    t = time.time(); out = []
    games = pickle.load(open(GAMES, "rb")); rng = random.Random(SEED0)
    pos = [(gi, ti) for gi, (traj, _) in enumerate(games) for ti in range(len(traj)) if traj[ti]["move"][0] == "play"]
    def tasks():
        for i in range(NPOS):
            gi, ti = pos[rng.randrange(len(pos))]; r = games[gi][0][ti]
            yield (i, dict(board=r["board"], blanks=r["blanks"], rack=r["rack"], unseen=r["unseen"], scores=r["scores"], bag=r["bag"], turn=r["turn"], game=(gi, ti)))
    print(f"{len(pos)} positions available; labelling {NPOS}", flush=True)
    with mp.Pool(NPROC, initializer=init) as pool:
        for j, r in enumerate(pool.imap_unordered(one, tasks(), chunksize=4)):
            if r is not None: out.append(r)
            if (j + 1) % 1000 == 0:
                print(f"{j+1}/{NPOS} positions {time.time()-t:.0f}s", flush=True)
                if (j + 1) % 20000 == 0: pickle.dump(out, open(f"data/simlabels_{TAG}.pkl", "wb"))
    pickle.dump(out, open(f"data/simlabels_{TAG}.pkl", "wb"))
    est = np.array([x["est"][0] for x in out]); eq = np.array([x["cands"][0][7] for x in out])
    print(f"DONE {len(out)} positions {time.time()-t:.0f}s; top-candidate: sim est mean {est.mean():.1f}, macondo equity mean {eq.mean():.1f}, MAE(equity vs sim) {np.abs(est-eq).mean():.1f} pts")
