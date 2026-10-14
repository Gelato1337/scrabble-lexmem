"""Self-play: net vs net with 1-ply search, many CPU processes. Writes data/games_{TAG}.pkl in our trajectory format.
Usage: CKPT=data/policy_X.pt USE_LEX=0 NPROC=60 TAG=sp1 PYTHONPATH=. python selfplay/selfplay.py NGAMES [seed0]"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
NG = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
E = os.environ.get; CKPT = E("CKPT"); USE_LEX = E("USE_LEX", "0") == "1"; LEXICON = E("LEXICON", "data/NWL23.txt"); NPROC = int(E("NPROC", 8)); TAG = E("TAG", "sp")
K = int(E("K", 6)); S = int(E("S", 6)); MIX = float(E("MIX", 0)); SEARCH = E("SEARCH", "1") == "1"; SIM = E("SIM", "0") == "1"
S2 = int(E("S2", 0)); ROLLOUT = E("ROLLOUT", "hasty"); CAND = E("CAND", "both"); VCKPT = E("VCKPT", ""); VLEX = E("VLEX", "0") == "1"; TEMP = float(E("TEMP", 1.0)); P_EXPLORE = float(E("P_EXPLORE", 0.1))
G = {}
def init():
    from lexmem.core import Lexicon
    from lexmem.policy import PolicyNet, MoveDataset
    lex = Lexicon(LEXICON); model = PolicyNet(lex, use_lex=USE_LEX)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False); model.eval()
    if USE_LEX: model.refresh_keys()
    G.update(lex=lex, model=model, helper=MoveDataset(lex, [], use_lex=USE_LEX))
    if SIM:
        from lexmem.macondo_gen import MacondoGen; G["mg"] = MacondoGen()
        if VCKPT:
            from lexmem.sim import ValueEval; G["vnet"] = ValueEval(VCKPT, lex, VLEX)
@torch.no_grad()
def sample_policy(model, helper, lex, board, blanks, cc, rack, unseen, scores, bag, turn, rng, temp):
    from lexmem.policy import collate_moves, MAXM
    from lexmem.core import legal_moves
    rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=bag, scores=scores, turn=turn)
    it = helper.build(rec, 0.0)
    if it is None: return None, 0.0
    b = collate_moves([it]); pl, v = model(b); pl = pl[0].float()
    mvs = legal_moves(lex, board, blanks, cc, rack); mvs.sort(key=lambda m: -m[0]); mvs = mvs[:MAXM]
    p = torch.softmax(pl[:len(mvs)] / max(temp, 1e-3), 0).numpy(); k = rng.choices(range(len(mvs)), weights=p)[0]
    return mvs[k], (float(v[0]), float(np.log(max(p[k], 1e-12))))
def one(g):
    from lexmem.core import FULL_BAG, empty_board, counts27, draw_tiles, cross_checks, apply_move, LET, VAL
    from lexmem.search import choose_move
    lex, model, helper = G["lex"], G["model"], G["helper"]; rng = random.Random(SEED0 + g)
    bag = [i for i in range(27) for _ in range(FULL_BAG[i])]; rng.shuffle(bag)
    board, blanks = empty_board(); racks = [counts27(draw_tiles(bag, 7, rng)), counts27(draw_tiles(bag, 7, rng))]
    scores = [0, 0]; zeros = 0; turn = 0; traj = []
    while True:
        me = turn % 2; rack = racks[me]; unseen = counts27(bag) + racks[1 - me]
        cc = cross_checks(lex, board, blanks)
        rec = dict(board=board.copy(), blanks=blanks.copy(), rack=rack.copy(), unseen=unseen.copy(), scores=(scores[me], scores[1 - me]), bag=len(bag), turn=turn, mover=me)
        if SIM:   # expert iteration: the rollout sim chooses; store candidates + estimates as soft targets
            from lexmem.sim import sim_choose
            explore = False
            out = sim_choose(model, helper, lex, G["mg"], board, blanks, rack, unseen, (scores[me], scores[1 - me]), len(bag), turn, rng, K=K, S=S, S2=S2, rollout=ROLLOUT, cand=CAND, return_all="raw", hval=G.get("vnet"))
            if out is None: mv = None
            else:
                mv, cands, est = out; mv = tuple(mv[:6])
                rec["cands"] = [tuple(m[1:6]) for m in cands]; rec["cand_est"] = [float(x) for x in est]
        elif SEARCH:
            explore = rng.random() < P_EXPLORE or turn < 2
            mv = choose_move(model, helper, lex, board, blanks, cc, rack, unseen, (scores[me], scores[1 - me]), len(bag), turn, rng, K=K, S=S, temp=TEMP if explore else 0.0, mix=MIX)
        else:   # policy-gradient mode: sample the bare policy at TEMP on every move; record the value baseline
            explore = False
            mv, vb = sample_policy(model, helper, lex, board, blanks, cc, rack, unseen, (scores[me], scores[1 - me]), len(bag), turn, rng, TEMP)
            if mv is not None: rec["v_base"], rec["logp_old"] = vb
        if mv is None:
            if len(bag) >= 7:   # exchange everything
                tiles = [i for i in range(27) for _ in range(rack[i])]; newt = draw_tiles(bag, 7, rng); bag.extend(tiles); rng.shuffle(bag)
                racks[me] = counts27(newt); rec.update(move=("exchange",), placed=[], drawn=newt)
            else: rec.update(move=("pass",), placed=[], drawn=[])
            zeros += 1
        else:
            s, r, c, d, L, widx = mv
            board, blanks, rack2, placed = apply_move(lex, board, blanks, rack, r, c, d, L, widx)
            letters = ""; dr, dc = ((0, 1), (1, 0))[d]
            for j in range(L):
                ch = LET[lex.POS[widx, j]]; letters += "." if rec["board"][r + dr * j, c + dc * j] >= 0 else (ch.lower() if any(p[0] == r + dr * j and p[1] == c + dc * j for p in placed) and blanks[r + dr * j, c + dc * j] else ch)
            drawn = draw_tiles(bag, 7 - int(rack2.sum()), rng); racks[me] = rack2 + counts27(drawn); scores[me] += s; zeros = 0
            rec.update(move=("play", r, c, d, L, letters, s), placed=placed, drawn=drawn, explore=explore)
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
            if (i + 1) % 100 == 0: print(f"{i+1} games {time.time()-t:.0f}s  mean score {np.mean([x[1] for x in games]):.0f}  mean turns {np.mean([len(x[0]) for x in games]):.1f}", flush=True)
            if (i + 1) % 500 == 0: pickle.dump(games, open(f"data/games_{TAG}.pkl", "wb"))
    pickle.dump(games, open(f"data/games_{TAG}.pkl", "wb"))
    sc = np.array([g[1] for g in games]); print(f"DONE {NG} games {time.time()-t:.0f}s; first-mover wins {np.mean(sc[:,0]>sc[:,1]):.3f}; mean score {sc.mean():.0f}; saved data/games_{TAG}.pkl")
