"""
Shared rules + lexicon machinery for the feasibility experiments.

Board: int8 15x15, -1 empty, 0..25 letter index. BLANKS: bool 15x15 (tile is a blank).
Lexicon: ENABLE (enable1.txt), lengths 2..15.

Provides
  - full premium layout, exact scoring incl. cross-words and bingo
  - cross-check sets per empty square (for both directions)
  - lane enumeration (all placements where a word could go)
  - legal move generation for a rack (exact: pattern + cross-checks + rack letters + blanks)
  - the STEP-2 LABEL: for a (rack, U, lane), every word fitting the lane whose remaining
    letters (after rack) are <= MAXMISS and available in U, weighted by
        P(a draw of n = 7 - kept tiles from U contains the missing letters) * score(word in lane)
    Exactly computable (multivariate hypergeometric).
"""
import numpy as np, random, math, os
from collections import defaultdict
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DIST = dict(A=9,B=2,C=2,D=4,E=12,F=2,G=3,H=2,I=9,J=1,K=1,L=4,M=2,N=6,O=8,
            P=2,Q=1,R=6,S=4,T=6,U=4,V=2,W=2,X=1,Y=2,Z=1)
VALS = dict(A=1,B=3,C=3,D=2,E=1,F=4,G=2,H=4,I=1,J=8,K=5,L=1,M=3,N=1,O=1,
            P=3,Q=10,R=1,S=1,T=1,U=1,V=4,W=4,X=8,Y=4,Z=10)
LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX = {c: i for i, c in enumerate(LET)}
BLANK = 26                      # index of blank in 27-vectors
VAL = np.array([VALS[c] for c in LET] + [0], dtype=np.int32)   # value by letter idx (blank=0)
FULL_BAG = np.array([DIST[c] for c in LET] + [2], dtype=np.int16)  # 27 counts
N = 15
CENTER = (7, 7)
MAXMISS = 3                     # label: words at most this many letters away

# ---------------------------------------------------------------- premiums
TW = {(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)}
DW = {(i,i) for i in (1,2,3,4,10,11,12,13)} | {(i,14-i) for i in (1,2,3,4,10,11,12,13)} | {(7,7)}
TL = {(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),(9,1),(9,5),(9,9),(9,13),(13,5),(13,9)}
DL = {(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),(6,2),(6,6),(6,8),(6,12),(7,3),(7,11),
      (8,2),(8,6),(8,8),(8,12),(11,0),(11,7),(11,14),(12,6),(12,8),(14,3),(14,11)}
LMULT = np.ones((N, N), dtype=np.int32); WMULT = np.ones((N, N), dtype=np.int32)
for (r, c) in TL: LMULT[r, c] = 3
for (r, c) in DL: LMULT[r, c] = 2
for (r, c) in TW: WMULT[r, c] = 3
for (r, c) in DW: WMULT[r, c] = 2
PREM = np.zeros((N, N), dtype=np.int64)   # 0 none,1 DL,2 TL,3 DW,4 TW  (for features)
for (r, c) in DL: PREM[r, c] = 1
for (r, c) in TL: PREM[r, c] = 2
for (r, c) in DW: PREM[r, c] = 3
for (r, c) in TW: PREM[r, c] = 4

# ---------------------------------------------------------------- lexicon
class Lexicon:
    def __init__(self, path=os.path.join(ROOT, "enable1.txt"), maxlen=15):
        words = [w.strip().upper() for w in open(path)]
        words = [w for w in words if 2 <= len(w) <= maxlen and w.isalpha()]
        self.words = words
        self.NW = len(words)
        self.WSET = set(words); self.WIDX = {w: i for i, w in enumerate(words)}
        self.W = np.zeros((self.NW, 26), dtype=np.int16)      # letter counts
        self.POS = np.full((self.NW, maxlen), 26, dtype=np.int64)   # letter idx per position, 26 pad
        self.LEN = np.zeros(self.NW, dtype=np.int64)
        for i, w in enumerate(words):
            self.LEN[i] = len(w)
            for j, ch in enumerate(w):
                self.W[i, IDX[ch]] += 1; self.POS[i, j] = IDX[ch]
        self.by_len = {L: np.where(self.LEN == L)[0] for L in range(2, maxlen + 1)}
        # per length: positions matrix and counts, for fast lane filtering
        self.POS_L = {L: self.POS[idx, :L] for L, idx in self.by_len.items()}
        self.W_L = {L: self.W[idx] for L, idx in self.by_len.items()}
        self.VAL_L = {L: VAL[self.POS_L[L]] for L in self.by_len}   # letter values per position
        self.maxlen = maxlen

    def __len__(self): return self.NW

# ---------------------------------------------------------------- board helpers
def empty_board():
    return np.full((N, N), -1, dtype=np.int8), np.zeros((N, N), dtype=bool)

def board_str(board):
    return "\n".join("".join("." if x < 0 else LET[x] for x in row) for row in board)

def cross_checks(lex, board, blanks):
    """For each empty square and each placement direction, the allowed letters (26 bool) and
       the cross-word base score (sum of existing letters' values, 0 for blanks) and whether
       a cross word exists. Returns dict dir -> (allow[N,N,26] bool, base[N,N] int, has[N,N] bool).
       dir 0 = word placed horizontally (cross words vertical), dir 1 = vertical."""
    out = {}
    for d, (dr, dc) in enumerate(((0, 1), (1, 0))):
        pdr, pdc = dc, dr           # perpendicular
        allow = np.ones((N, N, 26), dtype=bool); base = np.zeros((N, N), dtype=np.int32)
        has = np.zeros((N, N), dtype=bool)
        for r in range(N):
            for c in range(N):
                if board[r, c] >= 0: continue
                pre = []; ar, ac = r - pdr, c - pdc
                while 0 <= ar < N and 0 <= ac < N and board[ar, ac] >= 0:
                    pre.append((board[ar, ac], blanks[ar, ac])); ar -= pdr; ac -= pdc
                pre = pre[::-1]
                suf = []; ar, ac = r + pdr, c + pdc
                while 0 <= ar < N and 0 <= ac < N and board[ar, ac] >= 0:
                    suf.append((board[ar, ac], blanks[ar, ac])); ar += pdr; ac += pdc
                if not pre and not suf: continue
                has[r, c] = True
                base[r, c] = sum(0 if b else VAL[l] for l, b in pre + suf)
                ps = "".join(LET[l] for l, _ in pre); ss = "".join(LET[l] for l, _ in suf)
                allow[r, c] = [ (ps + ch + ss) in lex.WSET for ch in LET ]
        out[d] = (allow, base, has)
    return out

def lanes(board):
    """All (r, c, d, L) placements: start not preceded by a tile, end not followed by a tile,
       at least one empty square, <= 7 empties, touches a tile (or covers center on empty board)."""
    first = not (board >= 0).any()
    occ = board >= 0
    # adjacency map
    adj = np.zeros((N, N), dtype=bool)
    adj[1:, :] |= occ[:-1, :]; adj[:-1, :] |= occ[1:, :]; adj[:, 1:] |= occ[:, :-1]; adj[:, :-1] |= occ[:, 1:]
    out = []
    for d, (dr, dc) in enumerate(((0, 1), (1, 0))):
        for r in range(N):
            for c in range(N):
                pr, pc = r - dr, c - dc
                if 0 <= pr < N and 0 <= pc < N and occ[pr, pc]: continue
                nfix = 0; touch = False
                for L in range(1, N + 1):
                    er, ec = r + dr * (L - 1), c + dc * (L - 1)
                    if not (0 <= er < N and 0 <= ec < N): break
                    if occ[er, ec]: nfix += 1
                    if occ[er, ec] or adj[er, ec] or (first and (er, ec) == CENTER): touch = True
                    if L - nfix > 7: break
                    if L < 2: continue
                    nr, nc = er + dr, ec + dc
                    if 0 <= nr < N and 0 <= nc < N and occ[nr, nc]: continue
                    if nfix == L or not touch: continue
                    out.append((r, c, d, L))
    return out

def lane_info(board, blanks, cc, r, c, d, L):
    """Per-position arrays for a lane: fixed letter (-1 if empty), allowed (L,26) bool,
       letter mult, word mult, cross base, has cross, fixed values, is_blank."""
    dr, dc = ((0, 1), (1, 0))[d]
    allow_all, base_all, has_all = cc[d]
    rows = r + dr * np.arange(L); cols = c + dc * np.arange(L)
    fixed = board[rows, cols].astype(np.int64)
    empty = fixed < 0
    allow = np.ones((L, 26), dtype=bool)
    allow[empty] = allow_all[rows[empty], cols[empty]]
    return dict(rows=rows, cols=cols, fixed=fixed, empty=empty, allow=allow,
                lm=LMULT[rows, cols], wm=WMULT[rows, cols], base=base_all[rows, cols],
                has=has_all[rows, cols], fblank=blanks[rows, cols], prem=PREM[rows, cols])

def lane_candidates(lex, info, L):
    """Indices (into lexicon) of words that fit the pattern and cross-checks. Also returns
       local index into by_len[L]."""
    P = lex.POS_L[L]
    m = np.ones(P.shape[0], dtype=bool)
    fixed, empty, allow = info["fixed"], info["empty"], info["allow"]
    for j in range(L):
        if not empty[j]: m &= (P[:, j] == fixed[j])
        elif not allow[j].all(): m &= allow[j][P[:, j]]
    loc = np.where(m)[0]
    return lex.by_len[L][loc], loc

def lane_scores(lex, info, L, loc):
    """Score of each candidate word (local idx) placed in the lane, assuming no blanks used."""
    P = lex.POS_L[L][loc]; V = lex.VAL_L[L][loc]          # (n,L)
    empty = info["empty"]; lm = info["lm"]; wm = info["wm"]
    fixedval = np.where(info["fblank"], 0, V)               # blank tiles on board are worth 0
    lettersum = np.where(empty[None], V * lm[None], fixedval).sum(1)
    wmul = np.prod(np.where(empty, wm, 1))
    main = lettersum * wmul
    cross = (empty[None] & info["has"][None]) * (info["base"][None] + V * lm[None]) * wm[None]
    score = main + cross.sum(1)
    if empty.sum() == 7: score = score + 50
    return score.astype(np.int32)

def blank_penalty(info, letters, deficit):
    """Best blank designation for one word in a lane. letters: (L,) letter idx; deficit: (26,) blanks needed per letter.
    A blank on empty square j forfeits val*lm*(word mult of the play + word mult of j if it forms a cross word).
    Returns (total points forfeited, list of lane positions that take the blanks)."""
    empty, lm, wm, has = info["empty"], info["lm"], info["wm"], info["has"]
    wmul = int(np.prod(np.where(empty, wm, 1))); pen_all = VAL[letters] * lm * (wmul + has * wm); tot = 0; pos = []
    for l in np.where(deficit > 0)[0]:
        js = [j for j in range(len(letters)) if empty[j] and letters[j] == l]
        js.sort(key=lambda j: (pen_all[j], j)); take = js[:int(deficit[l])]; pos += take; tot += int(sum(pen_all[j] for j in take))
    return tot, pos

def designate(lex, board, rack27, r, c, d, L, widx):
    """Letters string for placing word widx ('.' = through tile, lowercase = blank), blanks on the best squares."""
    dr, dc = ((0, 1), (1, 0))[d]; rows = r + dr * np.arange(L); cols = c + dc * np.arange(L)
    fixed = board[rows, cols]; empty = fixed < 0; letters = lex.POS[widx, :L]
    pr, pc = (rows + dc, cols + dr), (rows - dc, cols - dr)   # perpendicular neighbours
    def occ(rr, cc_):
        ok = (rr >= 0) & (rr < N) & (cc_ >= 0) & (cc_ < N); out = np.zeros(L, dtype=bool)
        out[ok] = board[rr[ok], cc_[ok]] >= 0; return out
    info = dict(empty=empty, lm=LMULT[rows, cols], wm=WMULT[rows, cols], has=(occ(*pr) | occ(*pc)))
    need = np.zeros(26, dtype=np.int16)
    for j in range(L):
        if empty[j]: need[letters[j]] += 1
    deficit = np.maximum(need - rack27[:26], 0)
    bpos = set(blank_penalty(info, letters, deficit)[1]) if deficit.sum() > 0 else set()
    return "".join("." if not empty[j] else (LET[letters[j]].lower() if j in bpos else LET[letters[j]]) for j in range(L))

def need_counts(lex, info, L, loc):
    """Letters each candidate needs to place = word counts - fixed letters. (n,26)"""
    fixedc = np.zeros(26, dtype=np.int16)
    for f in info["fixed"]:
        if f >= 0: fixedc[f] += 1
    return np.maximum(lex.W_L[L][loc] - fixedc, 0)

# ---------------------------------------------------------------- draw probability
@lru_cache(maxsize=None)
def _comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0

@lru_cache(maxsize=500000)
def draw_prob(missing, u_missing, Ntot, n):
    """P(a uniform draw of n tiles from a bag of Ntot contains >= missing[i] copies of letter i),
       where u_missing[i] is the count of letter i in the bag. missing/u_missing are tuples."""
    if n <= 0: return 0.0 if sum(missing) > 0 else 1.0
    if sum(missing) > n: return 0.0
    for m, u in zip(missing, u_missing):
        if m > u: return 0.0
    rest = Ntot - sum(u_missing)
    tot = _comb(Ntot, n)
    if tot == 0: return 0.0
    # enumerate counts x_i in [missing_i, min(u_i, n)]
    def rec(i, used, prod):
        if i == len(missing):
            return prod * _comb(rest, n - used)
        s = 0.0
        for x in range(missing[i], min(u_missing[i], n - used) + 1):
            s += rec(i + 1, used + x, prod * _comb(u_missing[i], x))
        return s
    return rec(0, 0, 1.0) / tot

def reachable_label(lex, info, L, loc, rack27, unseen27, maxmiss=MAXMISS):
    """For candidates loc (already pattern+crosscheck filtered), compute
         from_rack (kept tiles), missing count, weight = P(draw) * score
       Returns arrays over loc: ok(bool: missing<=maxmiss and available), miss(int), P(float), score(int), weight(float)"""
    need = need_counts(lex, info, L, loc)                 # (n,26)
    rack = rack27[:26].astype(np.int16); rb = int(rack27[26])
    unseen = unseen27[:26].astype(np.int16); Ntot = int(unseen27[:26].sum())   # blanks in bag ignored for P
    from_rack = np.minimum(need, rack[None]).sum(1)
    deficit = np.maximum(need - rack[None], 0)            # (n,26) letters not on rack
    dsum = deficit.sum(1)
    n = need.sum(1)
    # rack blanks cover the rarest-in-U missing letters
    miss = dsum.copy()
    P = np.zeros(len(loc)); ok = np.zeros(len(loc), dtype=bool)
    score = lane_scores(lex, info, L, loc)
    avail_ok = (np.maximum(deficit - unseen[None], 0).sum(1) <= rb)
    for i in range(len(loc)):
        if not avail_ok[i]: continue
        d = deficit[i]
        if dsum[i] == 0:
            P[i] = 1.0; ok[i] = True; miss[i] = 0; continue
        letters = np.where(d > 0)[0].tolist()
        cnt = [int(d[l]) for l in letters]
        b = rb
        if b > 0:   # blanks cover letters with lowest availability first
            order = sorted(range(len(letters)), key=lambda k: unseen[letters[k]])
            for k in order:
                while b > 0 and cnt[k] > 0: cnt[k] -= 1; b -= 1
        m = sum(cnt)
        miss[i] = m
        if m > maxmiss: continue
        keep = int(from_rack[i]) + (rb - b)
        ndraw = 7 - keep
        if m == 0: P[i] = 1.0; ok[i] = True; continue
        ml = tuple(c for c in cnt if c > 0); ul = tuple(int(unseen[letters[k]]) for k in range(len(letters)) if cnt[k] > 0)
        P[i] = draw_prob(ml, ul, Ntot, ndraw)
        ok[i] = P[i] > 0
    weight = P * score
    return ok, miss, P, score, weight

# ---------------------------------------------------------------- move generation
def legal_moves(lex, board, blanks, cc, rack27, lane_list=None, max_lanes=None, rng=None):
    """All legal placements from the rack. Returns list of (score, r, c, d, L, word_idx, uses_blank_positions)."""
    rack = rack27[:26].astype(np.int16); rb = int(rack27[26])
    out = []
    ll = lane_list if lane_list is not None else lanes(board)
    if max_lanes and len(ll) > max_lanes:
        ll = (rng or random).sample(ll, max_lanes)
    for (r, c, d, L) in ll:
        info = lane_info(board, blanks, cc, r, c, d, L)
        gidx, loc = lane_candidates(lex, info, L)
        if len(loc) == 0: continue
        need = need_counts(lex, info, L, loc)
        deficit = np.maximum(need - rack[None], 0).sum(1)
        okm = deficit <= rb
        if not okm.any(): continue
        loc2 = loc[okm]; g2 = gidx[okm]
        sc = lane_scores(lex, info, L, loc2)
        if rb > 0:   # exact: each blank goes on the square where its letter would score least (= the best designation)
            df = np.maximum(need[okm] - rack[None], 0)
            for k in np.where(df.sum(1) > 0)[0]:
                sc[k] -= blank_penalty(info, lex.POS_L[L][loc2[k]], df[k])[0]
        for k in range(len(loc2)):
            out.append((int(sc[k]), r, c, d, L, int(g2[k])))
    return out

def apply_move(lex, board, blanks, rack27, r, c, d, L, widx):
    """Place word widx in lane; blanks go on the squares chosen by designate() (the exact best). Returns new rack27, placed list."""
    letters = designate(lex, board, rack27, r, c, d, L, widx)
    board = board.copy(); blanks = blanks.copy(); rack = rack27.copy()
    dr, dc = ((0, 1), (1, 0))[d]
    placed = []
    for j, ch in enumerate(letters):
        if ch == ".": continue
        rr, c2 = r + dr * j, c + dc * j; l = IDX[ch.upper()]
        board[rr, c2] = l
        if ch.islower():
            assert rack[BLANK] > 0
            rack[BLANK] -= 1; blanks[rr, c2] = True
        else:
            assert rack[l] > 0
            rack[l] -= 1
        placed.append((rr, c2, int(l)))
    return board, blanks, rack, placed

def counts27(tiles):
    v = np.zeros(27, dtype=np.int16)
    for t in tiles: v[t] += 1
    return v

def draw_tiles(bag, k, rng):
    """bag: list of tile idx. removes k random tiles, returns them."""
    k = min(k, len(bag))
    out = []
    for _ in range(k):
        i = rng.randrange(len(bag)); out.append(bag[i]); bag[i] = bag[-1]; bag.pop()
    return out

def play_game(lex, seed, temp=4.0, top_lanes=None, record=True):
    """Self-play one game with a softmax-over-score policy (HastyBot-lite, no leave values).
       Returns trajectory: list of dicts with board, blanks, rack (mover), unseen, scores, move, drawn."""
    rng = random.Random(seed)
    bag = [i for i in range(27) for _ in range(FULL_BAG[i])]; rng.shuffle(bag)
    board, blanks = empty_board()
    racks = [counts27(draw_tiles(bag, 7, rng)), counts27(draw_tiles(bag, 7, rng))]
    scores = [0, 0]; traj = []; scoreless = 0; turn = 0
    while True:
        me = turn % 2; rack = racks[me]
        cc = cross_checks(lex, board, blanks)
        unseen = counts27(bag) + racks[1 - me]
        mv = legal_moves(lex, board, blanks, cc, rack)
        rec = dict(board=board.copy(), blanks=blanks.copy(), rack=rack.copy(), unseen=unseen.copy(),
                   scores=(scores[me], scores[1 - me]), bag=len(bag), turn=turn, mover=me)
        if mv:
            sc = np.array([m[0] for m in mv], dtype=np.float64)
            p = np.exp((sc - sc.max()) / temp); p /= p.sum()
            k = rng.choices(range(len(mv)), weights=p)[0]
            s, r, c, d, L, widx = mv[k]
            board, blanks, rack, placed = apply_move(lex, board, blanks, rack, r, c, d, L, widx)
            drawn = draw_tiles(bag, 7 - int(rack.sum()), rng)
            rack = rack + counts27(drawn); racks[me] = rack; scores[me] += s
            rec.update(move=("play", r, c, d, L, widx, s), placed=placed, drawn=drawn)
            scoreless = 0
        else:
            if len(bag) >= 7:   # exchange everything
                tiles = [i for i in range(27) for _ in range(rack[i])]
                newt = draw_tiles(bag, 7, rng); bag.extend(tiles); rng.shuffle(bag)
                racks[me] = counts27(newt); rec.update(move=("exchange",), placed=[], drawn=newt)
            else:
                rec.update(move=("pass",), placed=[], drawn=[])
            scoreless += 1
        if record: traj.append(rec)
        turn += 1
        if racks[me].sum() == 0 and len(bag) == 0:
            other = racks[1 - me]; pen = int((other * VAL[:27]).sum())
            scores[me] += pen; scores[1 - me] -= pen; break
        if scoreless >= 6 or turn > 60: break
    return traj, scores
