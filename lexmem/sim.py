"""Monte-Carlo rollout search (Quackle-style; horizon = equity of the last ply) with the net as the rollout player and Macondo as the move generator.
For each of K candidate moves (top by the net's policy): for S sampled worlds (opponent rack + my draw, common across
candidates): opponent replies (net argmax or HastyBot top-equity), I follow up (net argmax); estimate = own score
- reply score + follow-up score, averaged. Bag empty -> the sampled world is exact."""
import numpy as np, torch, random
from lexmem.core import *
from lexmem.policy import collate_moves, MAXM

@torch.no_grad()
def policy_pick(model, helper, lex, positions, rollout):
    """positions: list of (board, blanks, rack, unseen, scores, bag, turn, moves). returns chosen move per position (or None).
       rollout='hasty' -> Macondo's first (top-equity) move; 'net' -> batched net argmax."""
    out = [None] * len(positions)
    if rollout == "hasty":
        for i, p in enumerate(positions): out[i] = p[7][0] if p[7] else None
        return out
    items = []; idx = []
    for i, (board, blanks, rack, unseen, scores, bag, turn, moves) in enumerate(positions):
        if not moves: continue
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=bag, scores=scores, turn=turn)
        it = helper.build(rec, 0.0, moves=moves[:MAXM])
        if it is None: continue
        items.append(it); idx.append(i)
    for j in range(0, len(items), 32):
        b = collate_moves(items[j:j + 32]); pl, v = model(b)
        for k, i in enumerate(idx[j:j + 32]):
            out[i] = positions[i][7][int(pl[k].float().argmax())]
    return out

def sim_choose(model, helper, lex, mg, board, blanks, rack, unseen, scores, bag, turn, rng, K=6, S=8, rollout="net", return_all=False, cand="net", S2=0, hval=None):
    """cand: 'net' = top-K by the net's policy, 'hasty' = top-K by Macondo equity. S2>0: successive halving — after S worlds
       keep the top 2 candidates and evaluate S2 more worlds on them."""
    moves = mg.gen(lex, board, blanks, rack, scores, 0, n=400)
    if not moves: return None
    if cand == "hasty": cands = moves[:K]
    else:
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=bag, scores=scores, turn=turn)
        it = helper.build(rec, 0.0, moves=moves[:MAXM])
        if it is None: return None
        with torch.no_grad(): pl, v = model(collate_moves([it]))
        pl = pl[0].float()[:min(len(moves), MAXM)]
        top = pl.topk(min(K, len(pl))).indices.tolist(); cands = [moves[k] for k in top]
        if cand == "both":
            keys = set((m[1], m[2], m[3], m[5]) for m in cands)
            for m in moves[:K]:
                if (m[1], m[2], m[3], m[5]) not in keys: cands.append(m); keys.add((m[1], m[2], m[3], m[5]))
    if len(cands) == 1: return (cands[0], cands, np.zeros(1)) if return_all == "raw" else cands[0]
    pool = [t for t in range(27) for _ in range(int(unseen[t]))]
    def make_worlds(n):
        ws = []
        for _ in range(n): rng.shuffle(pool); ws.append(list(pool))
        return ws
    tot = np.zeros(len(cands)); cnt = np.zeros(len(cands))
    sums = sim_eval(model, helper, lex, mg, board, blanks, rack, unseen, scores, turn, cands, make_worlds(S), rollout, hval)
    tot += sums; cnt += S
    if S2 > 0 and len(cands) > 2:
        keep = np.argsort(-(tot / cnt))[:2]
        sums2 = sim_eval(model, helper, lex, mg, board, blanks, rack, unseen, scores, turn, [cands[i] for i in keep], make_worlds(S2), rollout, hval)
        for j, i in enumerate(keep): tot[i] += sums2[j]; cnt[i] += S2
    est = tot / cnt
    best = int(np.argmax(est))
    if return_all == "raw": return cands[best], cands, est
    if return_all: return cands[best], list(zip([lex.words[m[5]] for m in cands], est.round(1).tolist(), cnt.astype(int).tolist()))
    return cands[best]

def sim_eval(model, helper, lex, mg, board, blanks, rack, unseen, scores, turn, cands, worlds, rollout, hval=None, record=None):
    """sum over worlds of (own score - opponent reply score + horizon value of my follow-up) per candidate.
       horizon = Macondo equity of the follow-up (score + leave table), or hval(list of post-follow-up states) -> points if given."""
    S = len(worlds); est = np.zeros(len(cands))
    # ply 1: opponent replies in every (candidate, world)
    P1 = []; meta = []
    for ci, m in enumerate(cands):
        s, r, c, d, L, widx = m[:6]
        b2, bl2, leave, placed = apply_move(lex, board, blanks, rack, r, c, d, L, widx)
        ndraw = 7 - int(leave.sum())
        for wi, w in enumerate(worlds):
            opp = counts27(w[:7]) if len(w) >= 7 else counts27(w)
            mydraw = counts27(w[7:7 + ndraw]) if len(w) > 7 else counts27([])
            opp_moves = mg.gen(lex, b2, bl2, opp, (scores[1], scores[0] + s), 0, n=400)
            P1.append((b2, bl2, opp, unseen - opp + leave, (scores[1], scores[0] + s), max(0, len(w) - 7), turn + 1, opp_moves))
            meta.append((ci, wi, s, b2, bl2, leave + mydraw, opp))
    replies = policy_pick(model, helper, lex, P1, rollout)
    if record is not None:   # future words: opponent replies (side 1), lane (r,c,d,L) on the post-move board
        for (ci, wi, *_), rep in zip(meta, replies):
            if rep is not None: record.append((ci, 1) + tuple(int(v) for v in rep[1:6]))
    # ply 2: my follow-up
    P2 = []; meta2 = []
    for (ci, wi, s, b2, bl2, myrack, opp), rep in zip(meta, replies):
        if rep is None: est[ci] += s / S; continue
        rs, rr, rc, rd, rL, rw = rep[:6]
        b3, bl3, opp_leave, _ = apply_move(lex, b2, bl2, opp, rr, rc, rd, rL, rw)
        my_moves = mg.gen(lex, b3, bl3, myrack, (scores[0] + s, scores[1] + rs), 0, n=400)
        P2.append((b3, bl3, myrack, unseen, (scores[0] + s, scores[1] + rs), 0, turn + 2, my_moves)); meta2.append((ci, s, rs))
    follow = policy_pick(model, helper, lex, P2, rollout)
    if record is not None:   # future words: my follow-ups (side 0)
        for (ci, _, _), f in zip(meta2, follow):
            if f is not None: record.append((ci, 0) + tuple(int(v) for v in f[1:6]))
    if hval is not None:   # learned horizon: follow-up score + V(state after the follow-up)
        states = []; owners = []
        for (ci, s, rs), f, p2 in zip(meta2, follow, P2):
            if f is None: est[ci] += (s - rs) / S; continue
            b3, bl3, myrack = p2[0], p2[1], p2[2]; fs, fr, fc, fd, fL, fw = f[:6]
            b4, bl4, leave4, _ = apply_move(lex, b3, bl3, myrack, fr, fc, fd, fL, fw)
            states.append((b4, leave4, p2[3], (p2[4][0] + fs, p2[4][1]), p2[5], p2[6] + 1)); owners.append((ci, s, rs, fs))
        vals = hval(states) if states else []
        for (ci, s, rs, fs), v in zip(owners, vals): est[ci] += (s - rs + fs + v) / S
        return est * S
    for (ci, s, rs), f in zip(meta2, follow):
        # horizon valuation: my follow-up counts at its EQUITY (score + leave value from Macondo's gen row), not raw score,
        # otherwise the sim rewards dumping tiles on the last ply
        est[ci] += (s - rs + ((f[7] if len(f) > 7 else f[0]) if f is not None else 0)) / S
    return est * S


class ValueEval:
    """wraps a trained ValueNet: states [(board, leave, unseen, scores, bag, turn)] -> value in points (future value of the leave on that board)."""
    def __init__(self, ckpt, lex, use_lex, nlane=6, retr=None):
        import train_value as tv   # ValueNet, hot_lanes, collate are defined there (module-level script; import guards below)
        self.tv = tv; self.lex = lex; self.use_lex = use_lex; self.nlane = nlane
        tv.USE_LEX = use_lex; tv.NLANE = nlane      # collate() reads these module globals
        self.model = tv.ValueNet(lex, use_lex)
        if use_lex: self.model.refresh_keys()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False); self.model.eval()
    def item(self, board, leave, unseen, scores, bag, turn):
        it = dict(board=board.astype(np.int64) + 1, rack=leave.astype(np.float32), unseen=unseen.astype(np.float32),
                  glob=np.array([bag / 100, (scores[0] - scores[1]) / 100, turn / 30], dtype=np.float32), target=np.float32(0), base=np.float32(0))
        if self.use_lex:
            from lexmem.policy import retrieval_mask, RET_LENS
            it["rmask"] = [retrieval_mask(self.lex, leave, unseen, L) for L in RET_LENS]
            it["lanes"] = self.tv.hot_lanes(self.lex, board, np.zeros_like(board, dtype=bool), leave, unseen, self.nlane)
        return it
    @torch.no_grad()
    def __call__(self, states):
        out = []
        for j in range(0, len(states), 64):
            items = [self.item(*s) for s in states[j:j + 64]]
            b = self.tv.collate(items); out += (self.model(b).float() * 100).tolist()
        return out

def vsearch_choose(vnet, model, helper, lex, mg, board, blanks, rack, unseen, scores, bag, turn, K=8, KS=24):
    """one-ply learned-equity chooser: candidates = union(policy top-K, score top-KS); pick argmax(score + V(post-move state))."""
    moves = mg.gen(lex, board, blanks, rack, scores, 0, n=400)
    if not moves: return None
    cands = list(moves[:1]); keys = set((m[1], m[2], m[3], m[5]) for m in cands)
    if model is not None:
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=bag, scores=scores, turn=turn)
        it = helper.build(rec, 0.0, moves=moves[:MAXM])
        if it is not None:
            with torch.no_grad(): pl, v = model(collate_moves([it]))
            for k in pl[0].float()[:min(len(moves), MAXM)].topk(min(K, len(moves))).indices.tolist():
                m = moves[k]
                if (m[1], m[2], m[3], m[5]) not in keys: cands.append(m); keys.add((m[1], m[2], m[3], m[5]))
    for m in sorted(moves, key=lambda m: -m[0])[:KS]:
        if (m[1], m[2], m[3], m[5]) not in keys: cands.append(m); keys.add((m[1], m[2], m[3], m[5]))
    states = []
    for m in cands:
        s, r, c, d, L, widx = m[:6]; b2, bl2, leave, _ = apply_move(lex, board, blanks, rack, r, c, d, L, widx)
        states.append((b2, leave, unseen, (scores[0] + s, scores[1]), max(0, int(unseen.sum()) - 7), turn + 1))
    vals = vnet(states); est = np.array([m[0] for m in cands]) + np.array(vals)
    return cands[int(np.argmax(est))]
