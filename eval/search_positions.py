"""Search vs teacher on held-out positions: does our search's choice move towards the teacher's as it gets more worlds?
prep:    python eval/search_positions.py prep TEST N
         -> data/searchpos_<TEST>.jsonl: N positions of data/games_<TEST>.pkl (teacher labels from prep/label_bot.py: SIMMING_BOT
         10 s, its top-16 sim candidates with win %), at most one per game (seed 0), where the teacher played tiles, has >= 2
         candidates and the bag holds >= 8 tiles (the search phase: the games hand off to Macondo below that); no openings (both
         orientations are equivalent there). Fields: i, cgp,
         turn (read by go/lexmem_search POSITIONS=...), g, ti, teacher [r, c, d, widx], cands [[r, c, d, widx]], est (win %).
analyze: python eval/search_positions.py analyze TEST TAG [TAG ...]
         For each run data/search_<TAG>.jsonl (go/lexmem_search with POSITIONS=data/searchpos_<TEST>.jsonl: the policy's top K
         moves by logit and their playout values in S shared worlds): how often the policy's top move and the search's choice
         equal the teacher's move, using the first S' worlds (S' = 32, 64, ... up to S) and the harness's rule (a candidate
         replaces the policy's top move only if its mean paired gain exceeds Z standard errors, Z = 0 and 1); the change against
         the policy's top move (paired over positions) and how many switches gain / lose the teacher's move; the teacher-
         estimated loss of the choice (teacher's best win % minus the choice's, choices outside its 16 candidates counted at its
         lowest listed win %: a lower bound)."""
import sys, json, random, pickle, numpy as np
from lexmem.core import LET, BLANK, N, Lexicon

def cgp(board, blanks, rack27, scores, zeros):   # as lexmem.harness HastyBot.cgp: the mover's rack only (what the teacher saw)
    rows = []
    for r in range(N):
        s = ""; run = 0
        for c in range(N):
            if board[r, c] < 0: run += 1
            else:
                if run: s += str(run); run = 0
                ch = LET[board[r, c]]; s += ch.lower() if blanks[r, c] else ch
        if run: s += str(run)
        rows.append(s)
    rk = "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(int(rack27[t])))
    return f"{'/'.join(rows)} {rk}/ {int(scores[0])}/{int(scores[1])} {zeros} lex NWL23;"

def prep(test, n):
    lex = Lexicon("data/NWL23.txt"); games = pickle.load(open(f"data/games_{test}.pkl", "rb")); rng = random.Random(0)
    def key(board, r, c, d, letters):
        dr, dc = ((0, 1), (1, 0))[d]
        word = "".join(LET[board[r + dr * j, c + dc * j]] if ch == "." else ch.upper() for j, ch in enumerate(letters))
        return None if word not in lex.WIDX else [r, c, d, lex.WIDX[word]]
    pool = []
    for gi, (traj, sc) in enumerate(games):
        ok, zeros = [], 0
        for ti, rec in enumerate(traj):
            m = rec["move"]
            if m[0] == "play" and rec.get("known", True) and rec["bag"] >= 8 and len(rec.get("cands", [])) >= 2 and (rec["board"] >= 0).any():
                t = key(rec["board"], m[1], m[2], m[3], m[5])
                if t is not None: ok.append((ti, zeros, t))
            zeros = zeros + 1 if m[0] != "play" else 0
        if ok: pool.append((gi, rng.choice(ok)))
    rng.shuffle(pool)
    with open(f"data/searchpos_{test}.jsonl", "w") as f:
        for i, (gi, (ti, zeros, t)) in enumerate(pool[:n]):
            rec = games[gi][0][ti]
            f.write(json.dumps(dict(i=i, cgp=cgp(rec["board"], rec["blanks"], rec["rack"], rec["scores"], zeros), turn=int(rec["turn"]), g=gi, ti=ti,
                                    teacher=t, cands=[[int(x) for x in (c[0], c[1], c[2], c[4])] for c in rec["cands"]],
                                    est=[float(e) for e in rec["cand_est"]])) + "\n")
    print(f"{min(n, len(pool))} positions from {len(games)} games -> data/searchpos_{test}.jsonl")

def pick(V, S, Z):   # the harness's rule (lexmem_search choose) on the first S worlds
    v = V[:, :S]; d = v[1:] - v[0]; m = d.mean(1); se = np.sqrt(np.maximum((d ** 2).mean(1) - m ** 2, 0) / S)
    bi, bm = 0, 0.0
    for ci in range(len(m)):
        if m[ci] > bm and m[ci] > Z * se[ci]: bi, bm = ci + 1, m[ci]
    return bi

def analyze(test, tags):
    pos = {p["i"]: p for p in map(json.loads, open(f"data/searchpos_{test}.jsonl"))}
    for tag in tags:
        runs = [json.loads(l) for l in open(f"data/search_{tag}.jsonl")]; runs = [o for o in runs if o["ntok"] > 0]
        S = max(len(o["v"][0]) for o in runs if o["v"])
        keys, T, est = [], [], []
        for o in runs:
            p = pos[o["i"]]; keys.append([tuple(c[1:4] + [c[5]]) if c[0] == 0 else None for c in o["c"]])
            T.append(tuple(p["teacher"])); est.append({tuple(c): e for c, e in zip(p["cands"], p["est"])})
        def stats(ch):   # ch[j] = chosen candidate index of position j
            agree = np.array([keys[j][ch[j]] == T[j] for j in range(len(runs))], float)
            loss = np.array([max(est[j].values()) - est[j].get(keys[j][ch[j]], min(est[j].values())) for j in range(len(runs))])
            inset = np.mean([keys[j][ch[j]] in est[j] for j in range(len(runs))])
            return agree, loss, inset
        a0, l0, in0 = stats([0] * len(runs))
        cover = np.mean([T[j] in keys[j] for j in range(len(runs))])
        print(f"== {tag}: {len(runs)} positions, K = {len(runs[0]['c'])}, S up to {S}; teacher's move among the policy's top K: {100 * cover:.1f} %")
        print(f"   policy top move: agrees with the teacher {100 * a0.mean():.1f} % ± {100 * a0.std() / np.sqrt(len(a0)):.1f}, "
              f"teacher-estimated loss {l0.mean():.2f} win-% points (in teacher's set {100 * in0:.1f} %)")
        for Z in (0.0, 1.0):
            for s in [x for x in (32, 64, 128, 256, 512, 1024, 2048) if x <= S]:
                ch = [pick(np.array(o["v"]), s, Z) if o["v"] else 0 for o in runs]
                a, l, inset = stats(ch); d = a - a0; sw = np.array([c != 0 for c in ch])
                gain = int(((a == 1) & (a0 == 0)).sum()); lose = int(((a == 0) & (a0 == 1)).sum())
                print(f"   search Z={Z:g} S={s:4d}: agrees {100 * a.mean():.1f} % ({100 * d.mean():+.1f} ± {100 * d.std() / np.sqrt(len(d)):.1f} vs policy), "
                      f"switches {100 * sw.mean():.1f} % (to teacher's move {gain}, away from it {lose}), loss {l.mean():.2f} ({(l - l0).mean():+.2f}), in set {100 * inset:.1f} %")

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(sys.argv[2], int(sys.argv[3]))
    else: analyze(sys.argv[2], sys.argv[3:])
