"""Parallel evaluation of a policy checkpoint vs HastyBot (CPU, one Macondo shell per worker).
Usage: CKPT=data/policy_imit_nolex.pt USE_LEX=0 NPROC=60 PYTHONPATH=. python eval/play_eval.py 2000 [seed0]"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
NG = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
CKPT = os.environ.get("CKPT", "data/policy_imit_nolex.pt"); USE_LEX = os.environ.get("USE_LEX", "0") == "1"
LEXICON = os.environ.get("LEXICON", "data/NWL23.txt"); NPROC = int(os.environ.get("NPROC", 8)); RETR = os.environ.get("RETR", "")
SEARCH = os.environ.get("SEARCH", "0") == "1"; SIM = os.environ.get("SIM", "0") == "1"; ROLLOUT = os.environ.get("ROLLOUT", "net"); CAND = os.environ.get("CAND", "net"); S2 = int(os.environ.get("S2", 0)); VSEARCH = os.environ.get("VSEARCH", "0") == "1"; VCKPT = os.environ.get("VCKPT", ""); VLEX = os.environ.get("VLEX", "0") == "1"; HVAL = os.environ.get("HVAL", "0") == "1"; K = int(os.environ.get("K", 6)); S = int(os.environ.get("S", 6)); TAGE = os.environ.get("TAGE", ""); MIX = float(os.environ.get("MIX", 0))
OPP = os.environ.get("OPP", "hasty"); OPP_SECS = float(os.environ.get("OPP_SECS", 10))   # hasty = Macondo shell; else a BotRequest code via bin/lexmem_bot
ENDGAME = int(os.environ.get("ENDGAME", 0)); EG_SECS = float(os.environ.get("EG_SECS", 10))   # 1: bag empty -> Macondo solver; 2: also 1 tile in the bag
OURS = os.environ.get("OURS", ""); OURS_SECS = float(os.environ.get("OURS_SECS", 10))   # a Macondo bot code plays our side instead of the net
G = {}
def bot_cpus():
    """2 cores per worker, by pool index, out of the cores this job may use."""
    cores = sorted(os.sched_getaffinity(0)); i = (mp.current_process()._identity or (1,))[0] - 1
    return {cores[(2 * i) % len(cores)], cores[(2 * i + 1) % len(cores)]}
def init():
    from lexmem.core import Lexicon
    from lexmem.policy import PolicyNet, MoveDataset
    from lexmem.harness import HastyBot, MacondoBot
    lex = Lexicon(LEXICON); model = PolicyNet(lex, use_lex=USE_LEX)
    sd = torch.load(CKPT, map_location="cpu"); model.load_state_dict(sd, strict=False); model.eval()
    if USE_LEX: model.refresh_keys()
    G.update(lex=lex, model=model, helper=MoveDataset(lex, [], use_lex=USE_LEX, exch=os.environ.get("EXCH", "0") == "1"), hasty=HastyBot() if OPP == "hasty" else MacondoBot(OPP, OPP_SECS, cpus=bot_cpus()))
    if OURS: G["ours"] = MacondoBot(OURS, OURS_SECS, cpus=bot_cpus())
    if ENDGAME: G["eg"] = MacondoBot("SIMMING_BOT", EG_SECS, cpus=bot_cpus())
    if SIM or VSEARCH:
        from lexmem.macondo_gen import MacondoGen; G["mg"] = MacondoGen()
    if VCKPT:
        from lexmem.sim import ValueEval; G["vnet"] = ValueEval(VCKPT, lex, VLEX)
def one(g):
    from lexmem.core import legal_moves
    from lexmem.policy import collate_moves, MAXM
    from lexmem.harness import play_vs_hasty
    lex, model, helper = G["lex"], G["model"], G["helper"]
    from lexmem.search import choose_move
    from lexmem.sim import sim_choose, vsearch_choose
    @torch.no_grad()
    def pol(lex_, board, blanks, cc, rack, unseen, rng):
        if VSEARCH:
            mv = vsearch_choose(G["vnet"], model, helper, lex_, G["mg"], board, blanks, rack, unseen, (0, 0), max(0, int(unseen.sum()) - 7), 0, K=K)
            return None if mv is None else tuple(mv[:6])
        if SIM:
            mv = sim_choose(model, helper, lex_, G["mg"], board, blanks, rack, unseen, (0, 0), max(0, int(unseen.sum()) - 7), 0, rng, K=K, S=S, rollout=ROLLOUT, cand=CAND, S2=S2, hval=G["vnet"] if HVAL else None)
            return None if mv is None else tuple(mv[:6])
        if SEARCH:
            return choose_move(model, helper, lex_, board, blanks, cc, rack, unseen, (0, 0), max(0, int(unseen.sum()) - 7), 0, rng, K=K, S=S, mix=MIX)
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=max(0, int(unseen.sum()) - 7), scores=(0, 0), turn=0)
        it = helper.build(rec, 0.0)
        if it is None: return None
        b = collate_moves([it])
        act = os.environ.get("ACT", "policy")
        if act in ("q", "qtop"):   # Track A nets: highest P(win) + spread/400 from the Q head (qtop: among the policy's top TOPK)
            pl, q, _ = model.heads(b); q = q[0].float(); u = torch.sigmoid(q[:, 1]) + q[:, 0] / 4
            if act == "qtop":
                top = pl[0].float().topk(min(int(os.environ.get("TOPK", 8)), u.shape[0])).indices; m_ = torch.full_like(u, -1e4); m_[top] = 0; u = u + m_
            k = int(u.argmax())
        else: pl, v = model(b); k = int(pl[0].argmax())
        m = it["_moves"][k]
        if m[3] == 2:   # exchange token (EXCH=1)
            from lexmem.policy import exchange_string; return exchange_string(rack, m[5])
        return m
    class Hybrid:   # the net until the end of the bag, then Macondo's endgame / pre-endgame solver (via SIMMING_BOT)
        wants_state = True
        def move(self, board, blanks, rack, scores, zeros):
            from lexmem.core import counts27, cross_checks, FULL_BAG, BLANK
            on = counts27(list(board[board >= 0])); on[BLANK] = int(blanks.sum())
            for (r_, c_) in zip(*np.where(blanks)): on[board[r_, c_]] -= 1
            unseen = FULL_BAG - on - rack
            if int(unseen.sum()) <= 6 + ENDGAME:   # unseen <= 7: bag empty; <= 8: one tile in the bag
                G["n_eg"] = G.get("n_eg", 0) + 1; return G["eg"].move(board, blanks, rack, scores, zeros)
            return pol(lex, board, blanks, cross_checks(lex, board, blanks), rack, unseen, random.Random(0))
    player = G["ours"] if OURS else (Hybrid() if ENDGAME else pol)
    spread, scores, log = play_vs_hasty(lex, player, G["hasty"], SEED0 + g, we_first=(g % 2 == 0))
    return spread
if __name__ == "__main__":
    t = time.time()
    with mp.Pool(NPROC, initializer=init) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(one, range(NG))):
            res.append(r)
            if (i + 1) % 100 == 0: a = np.array(res); print(f"{i+1} games: win {np.mean(a>0):.3f} spread {a.mean():+.1f} ± {a.std()/np.sqrt(len(a)):.1f}  {time.time()-t:.0f}s", flush=True)
    a = np.array(res); se = np.sqrt(np.mean(a > 0) * (1 - np.mean(a > 0)) / len(a))
    print(f"FINAL endgame={ENDGAME} {OURS + f'({OURS_SECS:g}s)' if OURS else os.path.basename(CKPT)} search={SEARCH} sim={SIM} vsearch={VSEARCH} hval={HVAL} vckpt={os.path.basename(VCKPT)} cand={CAND} rollout={ROLLOUT} K={K} S={S} S2={S2} mix={MIX} vs {OPP}{'' if OPP == 'hasty' else f'({OPP_SECS:g}s)'}: {NG} games, win rate {np.mean(a>0):.3f} ± {se:.3f}, ties {np.mean(a==0):.3f}, spread {a.mean():+.1f} ± {a.std()/np.sqrt(len(a)):.1f}, first-seat win {np.mean(a[0::2]>0):.3f} second-seat win {np.mean(a[1::2]>0):.3f}")
    pickle.dump(a, open(f"data/eval_{('bot_' + OURS) if OURS else os.path.basename(CKPT)[:-3]}{'_search' if SEARCH else ''}{'_sim' if SIM else ''}{'_vsearch' if VSEARCH else ''}{'_hval' if HVAL else ''}{TAGE}{'' if OPP == 'hasty' else '_vs_' + OPP}_{NG}.pkl", "wb"))
