"""Multiprocess replay of a Macondo autoplay directory into data/games_{tag}_train.pkl / _test.pkl."""
import sys, os, csv, glob, pickle, time, multiprocessing as mp
from collections import defaultdict
from lexmem.core import Lexicon
from lexmem.replay import replay_log
LEXICON = os.environ.get("LEXICON", "data/NWL23.txt")
d, tag = sys.argv[1], sys.argv[2]; ntest = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
tp = [p for p in glob.glob(d + "/*.txt") if "/games-" not in p][0]; gp = glob.glob(d + "/games-*.txt")[0]
lex = None
def init():
    global lex; lex = Lexicon(LEXICON)
def work(args):
    tpath, gpath = args
    g, n, bad = replay_log(lex, tpath, gpath)
    return g, n, bad
if __name__ == "__main__":
    t = time.time()
    # split the log into chunks by gameID
    header = None; rows = defaultdict(list)
    with open(tp) as f:
        rd = csv.reader(f); header = next(rd)
        for r in rd: rows[r[1]].append(r)
    gh = None; grows = {}
    with open(gp) as f:
        rd = csv.reader(f); gh = next(rd)
        for r in rd: grows[r[0]] = r
    gids = [g for g in rows if g in grows]; print(f"{len(gids)} games in log, {time.time()-t:.0f}s to read", flush=True)
    NP = int(os.environ.get("NPROC", 60)); chunks = [gids[i::NP] for i in range(NP)]
    os.makedirs("data/_chunks", exist_ok=True); jobs = []
    for i, ch in enumerate(chunks):
        tpath = f"data/_chunks/{tag}_{i}.txt"; gpath = f"data/_chunks/games-{tag}_{i}.txt"
        with open(tpath, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for g in ch: w.writerows(rows[g])
        with open(gpath, "w", newline="") as f:
            w = csv.writer(f); w.writerow(gh); w.writerows(grows[g] for g in ch)
        jobs.append((tpath, gpath))
    with mp.Pool(NP, initializer=init) as pool: res = pool.map(work, jobs)
    games = [g for r in res for g in r[0]]; n = sum(r[1] for r in res); bad = sum(r[2] for r in res)
    print(f"{len(games)} games, {sum(len(g[0]) for g in games)} positions, {n} plays, {bad} score mismatches, {time.time()-t:.0f}s", flush=True)
    pickle.dump(games[ntest:], open(f"data/games_{tag}_train.pkl", "wb")); pickle.dump(games[:ntest], open(f"data/games_{tag}_test.pkl", "wb"))
    for p in glob.glob("data/_chunks/*"): os.remove(p)
