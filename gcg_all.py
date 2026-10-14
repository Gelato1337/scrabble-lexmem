"""Cross-tables GCGs -> data/games_{tag}_train.pkl / _test.pkl (our trajectory format, see lexmem/gcg.py).
Duplicates (the same game annotated by both players) are collapsed on the move sequence; the test split is by game hash.
Usage: LEXICON=data/NWL23.txt NPROC=60 python gcg_all.py data/xt/gcg human 1000"""
import sys, os, glob, pickle, time, hashlib, collections, multiprocessing as mp
from lexmem.core import Lexicon
from lexmem.gcg import replay_gcg
LEXICON = os.environ.get("LEXICON", "data/NWL23.txt")
lex = None
def init():
    global lex; lex = Lexicon(LEXICON)
def work(path):
    try: traj, fin, st = replay_gcg(lex, open(path, errors="ignore").read())
    except Exception as e: return path, None, f"exception {type(e).__name__}"
    if fin is None: return path, None, st if isinstance(st, str) else st["cut"].split()[0]
    return path, (traj, fin), None
def key(traj):
    return hashlib.md5(repr([r["move"][:6] if r["move"][0] == "play" else r["move"][0] for r in traj]).encode()).hexdigest()
if __name__ == "__main__":
    d, tag = sys.argv[1], sys.argv[2]; ntest = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    paths = sorted(glob.glob(d + "/*.gcg")); t = time.time()
    with mp.Pool(int(os.environ.get("NPROC", 60)), initializer=init) as pool: res = pool.map(work, paths, chunksize=50)
    why = collections.Counter(r[2] for r in res if r[1] is None)
    games = {}
    for path, g, _ in res:
        if g is not None: games.setdefault(key(g[0]), g)
    ok = sum(r[1] is not None for r in res)
    order = sorted(games)                       # md5 order = a fixed pseudo-random order
    games = [games[k] for k in order]
    kp = sum(1 for tr, _ in games for r in tr if r["known"] and r["move"][0] == "play")
    print(f"{len(paths)} files: {ok} replayed, {ok - len(games)} duplicates, {len(games)} unique games, "
          f"{sum(len(g[0]) for g in games)} positions, {kp} known-rack plays (policy targets), {time.time()-t:.0f}s", flush=True)
    print("dropped:", why.most_common(), flush=True)
    pickle.dump(games[ntest:], open(f"data/games_{tag}_train.pkl", "wb")); pickle.dump(games[:ntest], open(f"data/games_{tag}_test.pkl", "wb"))
