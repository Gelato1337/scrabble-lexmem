"""Self-play game generation (multiprocess). Writes data/games_{tag}.pkl : list of (traj, scores)."""
import sys, os, pickle, time, multiprocessing as mp
from lexmem.core import Lexicon, play_game
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 400
TAG = sys.argv[2] if len(sys.argv) > 2 else "train"
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
lex = None
def init():
    global lex; lex = Lexicon()
def one(seed):
    return play_game(lex, seed)
if __name__ == "__main__":
    t = time.time()
    with mp.Pool(int(os.environ.get("NPROC", 14)), initializer=init) as pool:
        games = pool.map(one, range(SEED0, SEED0 + NG), chunksize=2)
    pickle.dump(games, open(f"data/games_{TAG}.pkl", "wb"))
    npos = sum(len(g[0]) for g in games)
    print(f"{NG} games, {npos} positions, {time.time()-t:.0f}s")
