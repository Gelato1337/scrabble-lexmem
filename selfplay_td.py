"""Track A self-play (TD-Gammon style): the net plays itself with no search; only game results will train it (train_td.py).
ACT=q: pick the move with the highest utility P(win) + spread/400 from the Q head (every legal move and exchange, one forward pass);
ACT=qtop: the same utility, restricted to the policy head's top TOPK moves (the policy as a prior over what to consider);
ACT=policy: the imitation policy head (warm-start round, before the Q head is trained).
Exploration: with prob EPS (and on the first OPEN_TURNS turns) sample softmax(utility / TEMP) instead of the argmax.
Luck bookkeeping: at each of a player's turns, K alternative racks they could have drawn after their previous move
(previous leave + a draw from the tiles that were unseen to them, consistent with the current board) -> rec["alt_racks"];
train_td.py prices the actual rack against them (draw luck) and uses their mean as the luck-free bootstrap value.
Usage: CKPT=... ACT=q NPROC=120 TAG=td_x_g1_0 python selfplay_td.py NGAMES SEED0 -> data/games_{TAG}.pkl"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
NG = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
E = os.environ.get; CKPT = E("CKPT", ""); ACT = E("ACT", "q"); EPS = float(E("EPS", 0.1)); TEMP = float(E("TEMP", 0.02)); PTEMP = float(E("PTEMP", 1.0))
TOPK = int(E("TOPK", 8)); OPEN_TURNS = int(E("OPEN_TURNS", 2)); K_ALT = int(E("K_ALT", 8)); TAG = E("TAG", "td"); NPROC = int(E("NPROC", 8))
G = {}
def init():
    from lexmem.core import Lexicon
    from lexmem.policy import PolicyNet, MoveDataset
    lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex)
    if CKPT: model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    model.eval(); G.update(lex=lex, model=model, helper=MoveDataset(lex, [], exch=True))
def utility(q):
    return torch.sigmoid(q[:, 1]) + q[:, 0] / 4      # q[:, 0] = spread / 100 -> 100 points ~ 0.25 win probability
@torch.no_grad()
def choose(rec, rng, explore):
    from lexmem.policy import collate_moves
    it = G["helper"].build(rec, 0.0)
    if it is None: return None
    b = collate_moves([it])
    if ACT == "policy":
        pl, _ = G["model"](b); s = pl[0].float(); T = PTEMP
    elif ACT == "qtop":
        pl, q, _ = G["model"].heads(b); s = utility(q[0].float()); T = TEMP
        top = pl[0].float().topk(min(TOPK, s.shape[0])).indices; mask = torch.full_like(s, -1e4); mask[top] = 0; s = s + mask
    else:
        q, _ = G["model"].qv(b); s = utility(q[0].float()); T = TEMP
    if explore:
        p = torch.softmax(s / T, 0).numpy().astype(np.float64); k = int(rng.choices(range(len(p)), weights=p)[0])
    else: k = int(s.argmax())
    return it["_moves"][k]
def one(g):
    from lexmem.core import FULL_BAG, BLANK, empty_board, counts27, draw_tiles, apply_move, LET, VAL
    lex = G["lex"]; rng = random.Random(SEED0 + g); nrng = np.random.default_rng(SEED0 + g)
    bag = [i for i in range(27) for _ in range(FULL_BAG[i])]; rng.shuffle(bag)
    board, blanks = empty_board(); racks = [counts27(draw_tiles(bag, 7, rng)), counts27(draw_tiles(bag, 7, rng))]
    scores = [0, 0]; zeros = 0; turn = 0; traj = []; last_leave = [None, None]
    while True:
        me = turn % 2; rack = racks[me]; unseen = counts27(bag) + racks[1 - me]
        rec = dict(board=board.copy(), blanks=blanks.copy(), rack=rack.copy(), unseen=unseen.astype(np.int16), scores=(scores[me], scores[1 - me]),
                   bag=len(bag), turn=turn, mover=me)
        if last_leave[me] is not None and K_ALT > 0:   # racks I could have drawn instead, given everything on the board now
            lv = last_leave[me]; need = int(rack.sum() - lv.sum())
            if need > 0:
                pool = (unseen + rack - lv).astype(np.int64); tiles = np.repeat(np.arange(27), pool)
                rec["alt_racks"] = np.stack([lv + np.bincount(nrng.choice(tiles, need, replace=False), minlength=27) for _ in range(K_ALT)]).astype(np.int16)
        explore = turn < OPEN_TURNS or rng.random() < EPS
        mv = choose(rec, rng, explore); rec["explore"] = explore
        if mv is None or (mv[3] == 2 and len(bag) < 7):
            rec.update(move=("pass",), placed=[]); zeros += 1; last_leave[me] = rack.copy()
        elif mv[3] == 2:   # exchange: keep mv[5]
            keep = np.asarray(mv[5], dtype=np.int16); out = rack - keep
            tiles = [t for t in range(27) for _ in range(int(out[t]))]; newt = draw_tiles(bag, len(tiles), rng); bag.extend(tiles); rng.shuffle(bag)
            racks[me] = keep + counts27(newt); rec.update(move=("exchange", tuple(int(x) for x in keep)), placed=[]); zeros += 1; last_leave[me] = keep.copy()
        else:
            s, r, c, d, L, widx = mv
            board, blanks, rack2, placed = apply_move(lex, board, blanks, rack, r, c, d, L, widx)
            letters = ""; dr, dc = ((0, 1), (1, 0))[d]
            for j in range(L):
                rr, c2 = r + dr * j, c + dc * j
                letters += "." if rec["board"][rr, c2] >= 0 else (LET[board[rr, c2]].lower() if blanks[rr, c2] else LET[board[rr, c2]])
            drawn = draw_tiles(bag, 7 - int(rack2.sum()), rng); racks[me] = rack2 + counts27(drawn); scores[me] += s; zeros = 0
            rec.update(move=("play", r, c, d, L, letters, s), placed=placed); last_leave[me] = rack2.copy()
        traj.append(rec); turn += 1
        if racks[me].sum() == 0 and len(bag) == 0:
            pen = int((racks[1 - me] * VAL[:27]).sum()); scores[me] += pen; scores[1 - me] -= pen; break
        if zeros >= 6 or turn > 80: break
    return traj, tuple(scores)
if __name__ == "__main__":
    t = time.time(); games = []
    with mp.Pool(NPROC, initializer=init) as pool:
        for i, g in enumerate(pool.imap_unordered(one, range(NG))):
            games.append(g)
            if (i + 1) % 200 == 0:
                sc = np.array([x[1] for x in games]); print(f"{i+1} games {time.time()-t:.0f}s  mean score {sc.mean():.0f}  turns {np.mean([len(x[0]) for x in games]):.1f}  "
                                                          f"exch {np.mean([r['move'][0] == 'exchange' for x in games for r in x[0]]):.3f}", flush=True)
    pickle.dump(games, open(f"data/games_{TAG}.pkl", "wb"))
    sc = np.array([g[1] for g in games])
    print(f"DONE {NG} games {time.time()-t:.0f}s ({(time.time()-t)/NG*NPROC:.1f} s/game/proc); mean score {sc.mean():.0f}; first-mover wins {np.mean(sc[:,0]>sc[:,1]):.3f}; saved data/games_{TAG}.pkl", flush=True)
