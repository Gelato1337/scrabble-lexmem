"""Track A v2 self-play: the policy plays itself (sampled at PTEMP; no Q argmax). Only game results train it (train_v2.py).
Branching (common random numbers): at a fraction P_BRANCH of positions, the chosen move a and an alternative b (sampled from the
policy's top BK moves, excluding a) are each played out to the end of the game with the greedy policy, from the same bag order
(per-player priority orders over tile instances, shared by both branches), so the draws are shared wherever the bag allows. rec["branch"] = (b, dU) with
dU = U(a) - U(b), U = win (1 / 0.5 / 0) + final spread / 400, from the mover's view: a low-noise comparison of two moves.
Usage: CKPT=... PTEMP=1.0 P_BRANCH=0.04 NPROC=124 TAG=v2_w_g1_0 python selfplay_v2.py NGAMES SEED0"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
NG = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
E = os.environ.get; CKPT = E("CKPT", ""); PTEMP = float(E("PTEMP", 1.0)); P_BRANCH = float(E("P_BRANCH", 0.04)); BK = int(E("BK", 6))
TAG = E("TAG", "v2"); NPROC = int(E("NPROC", 8))
G = {}
def init():
    from lexmem.core import Lexicon
    from lexmem.policy import PolicyNet, MoveDataset
    lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex)
    if CKPT: model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    model.eval(); G.update(lex=lex, model=model, helper=MoveDataset(lex, [], exch=True))
@torch.no_grad()
def logits(board, blanks, rack, unseen, scores, bag, turn):
    from lexmem.policy import collate_moves
    rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, scores=scores, bag=bag, turn=turn)
    it = G["helper"].build(rec, 0.0)
    if it is None: return None, None
    pl, _ = G["model"](collate_moves([it])); return it["_moves"], pl[0].float()
def apply(st, me, mv, draw):
    """apply move mv for player me to state st (dict, mutated). draw(k) -> list of tiles. returns record of the move."""
    from lexmem.core import apply_move, counts27, LET
    lex = G["lex"]; rack = st["racks"][me]
    if mv is None or (mv[3] == 2 and len(st["bag"]) < 7):
        st["zeros"] += 1; return ("pass",)
    if mv[3] == 2:
        keep = np.asarray(mv[5], dtype=np.int16); out = rack - keep; tiles = [t for t in range(27) for _ in range(int(out[t]))]
        newt = draw(len(tiles))
        if "put_back" in st: st["put_back"](tiles)
        else: st["bag"][:0] = tiles
        st["racks"][me] = keep + counts27(newt); st["zeros"] += 1; return ("exchange", tuple(int(x) for x in keep))
    s, r, c, d, L, widx = mv; before = st["board"]
    st["board"], st["blanks"], rack2, placed = apply_move(lex, st["board"], st["blanks"], rack, r, c, d, L, widx)
    letters = ""; dr, dc = ((0, 1), (1, 0))[d]
    for j in range(L):
        rr, c2 = r + dr * j, c + dc * j
        letters += "." if before[rr, c2] >= 0 else (LET[st["board"][rr, c2]].lower() if st["blanks"][rr, c2] else LET[st["board"][rr, c2]])
    st["racks"][me] = rack2 + counts27(draw(7 - int(rack2.sum()))); st["scores"][me] += s; st["zeros"] = 0
    return ("play", r, c, d, L, letters, s)
def finished(st, me):
    from lexmem.core import VAL
    if st["racks"][me].sum() == 0 and len(st["bag"]) == 0:
        pen = int((st["racks"][1 - me] * VAL[:27]).sum()); st["scores"][me] += pen; st["scores"][1 - me] -= pen; return True
    return st["zeros"] >= 6 or st["turn"] > 80
def play_out(st, first_move, mover, prio):
    """mover plays first_move, then both sides play the greedy policy to the end. Draws follow per-player priority orders over
    tile instances (prio[p] = list of (tile, copy)): a player takes the first instances still in the bag, so both branches give
    each player the same tiles wherever the bag contents agree (common random numbers that survive different move lengths)."""
    from lexmem.core import counts27
    bagc = counts27(st["bag"]).astype(np.int64)
    def sync(): st["bag"] = [t for t in range(27) for _ in range(int(bagc[t]))]
    def draw_for(p):
        def draw(k):
            out = []
            for (t, j) in prio[p]:
                if len(out) >= k or bagc.sum() == 0: break
                if j < bagc[t]: out.append(t); bagc[t] -= 1
            sync(); return out
        return draw
    def put_back(tiles):
        for t in tiles: bagc[t] += 1
        sync()
    st["put_back"] = put_back
    me = mover; apply(st, me, first_move, draw_for(me)); st["turn"] += 1
    while not finished(st, me):
        me = st["turn"] % 2; rack = st["racks"][me]; unseen = counts27(st["bag"]) + st["racks"][1 - me]
        mvs, pl = logits(st["board"], st["blanks"], rack, unseen, (st["scores"][me], st["scores"][1 - me]), len(st["bag"]), st["turn"])
        apply(st, me, None if mvs is None else mvs[int(pl.argmax())], draw_for(me)); st["turn"] += 1
    sp = st["scores"][mover] - st["scores"][1 - mover]
    return (1.0 if sp > 0 else 0.5 if sp == 0 else 0.0) + sp / 400
def one(g):
    from lexmem.core import FULL_BAG, empty_board, counts27, draw_tiles
    rng = random.Random(SEED0 + g); brng = random.Random(10 ** 9 + SEED0 + g)
    bag = [i for i in range(27) for _ in range(FULL_BAG[i])]; rng.shuffle(bag)
    board, blanks = empty_board()
    st = dict(board=board, blanks=blanks, racks=[counts27(draw_tiles(bag, 7, rng)), counts27(draw_tiles(bag, 7, rng))], scores=[0, 0], zeros=0, turn=0, bag=bag)
    traj = []; nb = 0
    while True:
        me = st["turn"] % 2; rack = st["racks"][me]; unseen = counts27(st["bag"]) + st["racks"][1 - me]
        rec = dict(board=st["board"].copy(), blanks=st["blanks"].copy(), rack=rack.copy(), unseen=unseen.astype(np.int16),
                   scores=(st["scores"][me], st["scores"][1 - me]), bag=len(st["bag"]), turn=st["turn"], mover=me)
        mvs, pl = logits(st["board"], st["blanks"], rack, unseen, rec["scores"], len(st["bag"]), st["turn"])
        mv = None
        if mvs is not None:
            p = torch.softmax(pl / PTEMP, 0).numpy().astype(np.float64); k = int(rng.choices(range(len(p)), weights=p)[0]); mv = mvs[k]
            rec["logp_old"] = float(np.log(max(p[k], 1e-12)))
            if len(mvs) > 1 and brng.random() < P_BRANCH:   # counterfactual pair with shared future draws
                top = [i for i in pl.topk(min(BK + 1, len(mvs))).indices.tolist() if i != k][:BK]
                if top:
                    w = torch.softmax(pl[top] / PTEMP, 0).numpy().astype(np.float64); b = top[int(brng.choices(range(len(top)), weights=w)[0])]
                    inst = [(t, j) for t in range(27) for j in range(int(FULL_BAG[t]))]; prio = []
                    for _ in range(2): o = list(inst); brng.shuffle(o); prio.append(o)
                    def copy_state(): return dict(board=st["board"].copy(), blanks=st["blanks"].copy(), racks=[x.copy() for x in st["racks"]],
                                                  scores=list(st["scores"]), zeros=st["zeros"], turn=st["turn"], bag=list(st["bag"]))
                    rec["branch"] = (mvs[b], play_out(copy_state(), mv, me, prio) - play_out(copy_state(), mvs[b], me, prio)); nb += 1
        def draw(k): return draw_tiles(st["bag"], k, rng)
        rec["move"] = apply(st, me, mv, draw); traj.append(rec); st["turn"] += 1
        if finished(st, me): break
    return traj, tuple(st["scores"]), nb
if __name__ == "__main__":
    t = time.time(); games = []; nbr = 0
    with mp.Pool(NPROC, initializer=init) as pool:
        for i, (tr, sc, nb) in enumerate(pool.imap_unordered(one, range(NG))):
            games.append((tr, sc)); nbr += nb
            if (i + 1) % 200 == 0:
                s = np.array([x[1] for x in games]); print(f"{i+1} games {time.time()-t:.0f}s  mean score {s.mean():.0f}  branches {nbr}", flush=True)
    pickle.dump(games, open(f"data/games_{TAG}.pkl", "wb"))
    d = np.array([r["branch"][1] for tr, _ in games for r in tr if "branch" in r])
    print(f"DONE {NG} games {time.time()-t:.0f}s ({(time.time()-t)/NG*NPROC:.1f} s/game/proc); {len(d)} branches, dU mean {d.mean() if len(d) else 0:+.3f} "
          f"sd {d.std() if len(d) else 0:.3f}, chosen move better in {np.mean(d > 0) if len(d) else 0:.3f}; saved data/games_{TAG}.pkl", flush=True)
