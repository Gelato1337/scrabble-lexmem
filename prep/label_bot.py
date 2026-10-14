"""Relabel whole games with a Macondo bot's choice (teacher), for imitation of a stronger player than HastyBot.
Positions come from data/games_{SRC}.pkl (HastyBot self-play; both racks known). Every position of games [G0, G1) is sent to
the teacher (bin/lexmem_bot, one process per worker pinned to its own cores); the record's move is replaced by the teacher's
play (known=True) or, for an exchange/pass/unparseable answer, kept but marked known=False (no policy target).
The game continuation (and so the value targets) stays HastyBot's.
CANDS=K also stores the sim's top-K candidates as soft targets: rec["cands"] = [(r,c,d,L,widx)], rec["cand_est"] = win % (x100),
rec["cand_eq"], rec["cand_iters"]; positions decided without a sim (endgame / pre-endgame) get the teacher move alone (est 100).
Usage: TEACHER=SIMMING_BOT SECS=10 NPROC=64 SRC=hasty_train PYTHONPATH=. python prep/label_bot.py G0 G1 TAG  -> data/games_{TAG}.pkl"""
import os, sys, time, pickle, multiprocessing as mp, numpy as np
from lexmem.core import *
from lexmem.replay import score_placement
from lexmem.harness import MacondoBot, macondo_play_to_letters
E = os.environ.get
TEACHER = E("TEACHER", "SIMMING_BOT"); SECS = float(E("SECS", 10)); SRC = E("SRC", "hasty_train"); CPW = int(E("CORES_PER_WORKER", 2))
G = {}
def init():
    cores = sorted(os.sched_getaffinity(0)); i = (mp.current_process()._identity or (1,))[0] - 1
    mine = {cores[(CPW * i + k) % len(cores)] for k in range(CPW)}; os.sched_setaffinity(0, mine)
    G.update(lex=Lexicon(E("LEXICON", "data/NWL23.txt")), bot=MacondoBot(TEACHER, SECS, cpus=mine))
def to_cand(lex, board, act):
    r, c, d, letters = act; dr, dc = ((0, 1), (1, 0))[d]
    if not all(0 <= r + dr * j < N and 0 <= c + dc * j < N for j in range(len(letters))): return None
    word = "".join(LET[board[r + dr * j, c + dc * j]] if ch == "." else ch.upper() for j, ch in enumerate(letters))
    w = lex.WIDX.get(word)
    return None if w is None else (r, c, d, len(letters), w)
def work(args):
    key, board, blanks, rack, scores, zeros, hasty = args
    lex, bot = G["lex"], G["bot"]
    play = bot.move(board, blanks, rack, scores, zeros); act = macondo_play_to_letters(play)
    if not isinstance(act, tuple): return key, None, play
    r, c, d, letters = act
    try: s, _ = score_placement(lex, board, blanks, cross_checks(lex, board, blanks), r, c, d, letters)
    except AssertionError: return key, None, play
    mv = ("play", r, c, d, len(letters), letters, s)
    if int(E("CANDS", 0)) > 0:
        cands, est, eq, it = [], [], [], []
        for wp, e, n, ign, p in bot.last_cands:
            a = macondo_play_to_letters(p); k = to_cand(lex, board, a) if isinstance(a, tuple) else None
            if k is not None and k not in cands: cands.append(k); est.append(100 * wp); eq.append(e); it.append(n)
        if not cands:
            k = to_cand(lex, board, act)
            if k is not None: cands, est, eq, it = [k], [100.0], [0.0], [0]
        if not cands: return key, None, play    # no usable soft target -> no policy target
        mv = mv + (dict(cands=cands, cand_est=est, cand_eq=eq, cand_iters=it),)
    return key, mv, play
if __name__ == "__main__":
    g0, g1, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; t = time.time()
    games = pickle.load(open(f"data/games_{SRC}.pkl", "rb"))[g0:g1]
    jobs = []
    for gi, (traj, sc) in enumerate(games):
        zeros = 0
        for ti, rec in enumerate(traj):
            if rec["bag"] > int(E("MAX_BAG", 999)): zeros = zeros + 1 if rec["move"][0] != "play" else 0; continue   # MAX_BAG: relabel late positions only
            jobs.append(((gi, ti), rec["board"], rec["blanks"], rec["rack"], tuple(int(x) for x in rec["scores"]), zeros, rec["move"]))
            zeros = zeros + 1 if rec["move"][0] != "play" else 0
    print(f"{len(games)} games, {len(jobs)} positions, teacher {TEACHER} {SECS:g}s", flush=True)
    n = agree = nplay = 0
    with mp.Pool(int(E("NPROC", 64)), initializer=init) as pool:
        for (gi, ti), mv, raw in pool.imap_unordered(work, jobs, chunksize=1):
            rec = games[gi][0][ti]; h = rec["move"]; rec.setdefault("hasty_move", h); rec["teacher_raw"] = raw
            if mv is None: rec["known"] = False
            else:
                if len(mv) > 7: rec.update(mv[7]); mv = mv[:7]
                nplay += 1; rec["known"] = True; rec["move"] = mv
                agree += h[0] == "play" and h[1:4] == mv[1:4] and h[5].upper() == mv[5].upper()
            n += 1
            if n % 2000 == 0: print(f"{n}/{len(jobs)} {time.time()-t:.0f}s  teacher plays {nplay/n:.3f}  agrees with HastyBot {agree/max(nplay,1):.3f}", flush=True)
    pickle.dump(games, open(f"data/games_{tag}.pkl", "wb"))
    print(f"DONE {n} positions, {nplay} teacher plays, agreement with HastyBot {agree/max(nplay,1):.3f}, {time.time()-t:.0f}s", flush=True)
