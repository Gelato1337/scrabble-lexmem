"""1-ply expectimax over exact futures: candidate moves from the policy, each scored by the position-only value head
from the opponent's view over S sampled opponent racks. Returns the chosen legal move (score, r, c, d, L, widx)."""
import numpy as np, torch, random
from lexmem.core import *
from lexmem.policy import collate_moves, MAXM

def opp_view(lex, board, blanks, r, c, d, widx, rack, unseen, my_score_after, opp_score, opp_racks):
    """apply the move exactly; return position items from the opponent's view for the GIVEN sampled opponent racks
       (common random numbers: every candidate is evaluated against the same racks)."""
    board2, blanks2, rack2, placed = apply_move(lex, board, blanks, rack, r, c, d, len(lex.words[widx]), widx)
    n_bag_after = max(0, int(unseen.sum()) - 7)
    return [(board2, opp, unseen - opp + rack2, (opp_score, my_score_after), n_bag_after) for opp in opp_racks]

def sample_opp_racks(unseen, rng, S):
    pool = [t for t in range(27) for _ in range(int(unseen[t]))]; out = []
    for _ in range(S):
        rng.shuffle(pool); out.append(counts27(pool[:7]))
    return out

@torch.no_grad()
def choose_move(model, helper, lex, board, blanks, cc, rack, unseen, scores, bag, turn, rng, K=6, S=6, temp=0.0, return_all=False, mix=0.0):
    """mix: weight of the policy prior. final = value_estimate(spread pts) + mix * (policy log-prob) ; mix=0 -> pure value search."""
    rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=bag, scores=scores, turn=turn)
    it = helper.build(rec, 0.0)
    if it is None: return None
    b = collate_moves([it]); pl, v = model(b); pl = pl[0].float()
    mvs = legal_moves(lex, board, blanks, cc, rack); mvs.sort(key=lambda m: -m[0]); mvs = mvs[:MAXM]
    if temp > 0:   # exploration: sample from the policy
        p = torch.softmax(pl / temp, 0).numpy(); k = rng.choices(range(len(mvs)), weights=p[:len(mvs)])[0]; return mvs[k]
    top = pl.topk(min(K, len(mvs))).indices.tolist()
    cands = [mvs[k] for k in top]
    items = []; owner = []; opp_racks = sample_opp_racks(unseen, rng, S)
    for ci, (s, r, c, d, L, widx) in enumerate(cands):
        for pos in opp_view(lex, board, blanks, r, c, d, widx, rack, unseen, scores[0] + s, scores[1], opp_racks):
            board2, opp, opp_unseen, sc, nb = pos
            items.append(helper.build_pos(board2, opp, opp_unseen, sc, nb, turn + 1)); owner.append(ci)
    bb = collate_moves(items); _, vals = model(bb, with_moves=False); vals = vals.float().numpy()
    est = np.zeros(len(cands))
    # value head = expected spread CHANGE over the next plies from the opponent's view (in /100). My estimate for a candidate:
    #   its own score  -  100 * E[opponent's future gain]
    for ci in range(len(cands)): est[ci] = cands[ci][0] - 100 * vals[np.array(owner) == ci].mean()
    if mix > 0: est = est + mix * torch.log_softmax(pl, 0)[top].numpy()
    best = int(np.argmax(est))
    if return_all: return cands[best], list(zip(cands, est.tolist(), pl[top].tolist()))
    return cands[best]
