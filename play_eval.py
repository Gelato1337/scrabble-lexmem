"""Parallel evaluation of a policy checkpoint vs HastyBot (CPU, one Macondo shell per worker).
Usage: CKPT=data/policy_imit_nolex.pt USE_LEX=0 NPROC=60 python play_eval.py 2000 [seed0]"""
import os, sys, time, pickle, random, numpy as np, torch, multiprocessing as mp
torch.set_num_threads(1)
NG = int(sys.argv[1]); SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
CKPT = os.environ.get("CKPT", "data/policy_imit_nolex.pt"); USE_LEX = os.environ.get("USE_LEX", "0") == "1"
LEXICON = os.environ.get("LEXICON", "data/NWL23.txt"); NPROC = int(os.environ.get("NPROC", 8)); RETR = os.environ.get("RETR", "")
SEARCH = os.environ.get("SEARCH", "0") == "1"; SIM = os.environ.get("SIM", "0") == "1"; ROLLOUT = os.environ.get("ROLLOUT", "net"); CAND = os.environ.get("CAND", "net"); S2 = int(os.environ.get("S2", 0)); VSEARCH = os.environ.get("VSEARCH", "0") == "1"; VCKPT = os.environ.get("VCKPT", ""); VLEX = os.environ.get("VLEX", "0") == "1"; HVAL = os.environ.get("HVAL", "0") == "1"; K = int(os.environ.get("K", 6)); S = int(os.environ.get("S", 6)); TAGE = os.environ.get("TAGE", ""); MIX = float(os.environ.get("MIX", 0))
G = {}
def init():
    from lexmem.core import Lexicon
    from lexmem.policy import PolicyNet, MoveDataset
    from harness import HastyBot
    lex = Lexicon(LEXICON); model = PolicyNet(lex, use_lex=USE_LEX)
    sd = torch.load(CKPT, map_location="cpu"); model.load_state_dict(sd, strict=False); model.eval()
    if USE_LEX: model.refresh_keys()
    G.update(lex=lex, model=model, helper=MoveDataset(lex, [], use_lex=USE_LEX), hasty=HastyBot())
    if SIM or VSEARCH:
        from lexmem.macondo_gen import MacondoGen; G["mg"] = MacondoGen()
    if VCKPT:
        from lexmem.sim import ValueEval; G["vnet"] = ValueEval(VCKPT, lex, VLEX)
def one(g):
    from lexmem.core import legal_moves
    from lexmem.policy import collate_moves, MAXM
    from harness import play_vs_hasty
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
        b = collate_moves([it]); pl, v = model(b); k = int(pl[0].argmax())
        mvs = legal_moves(lex_, board, blanks, cc, rack); mvs.sort(key=lambda m: -m[0]); return mvs[:MAXM][k]
    spread, scores, log = play_vs_hasty(lex, pol, G["hasty"], SEED0 + g, we_first=(g % 2 == 0))
    return spread
if __name__ == "__main__":
    t = time.time()
    with mp.Pool(NPROC, initializer=init) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(one, range(NG))):
            res.append(r)
            if (i + 1) % 100 == 0: a = np.array(res); print(f"{i+1} games: win {np.mean(a>0):.3f} spread {a.mean():+.1f} ± {a.std()/np.sqrt(len(a)):.1f}  {time.time()-t:.0f}s", flush=True)
    a = np.array(res); se = np.sqrt(np.mean(a > 0) * (1 - np.mean(a > 0)) / len(a))
    print(f"FINAL {os.path.basename(CKPT)} search={SEARCH} sim={SIM} vsearch={VSEARCH} hval={HVAL} vckpt={os.path.basename(VCKPT)} cand={CAND} rollout={ROLLOUT} K={K} S={S} S2={S2} mix={MIX} vs HastyBot: {NG} games, win rate {np.mean(a>0):.3f} ± {se:.3f}, ties {np.mean(a==0):.3f}, spread {a.mean():+.1f} ± {a.std()/np.sqrt(len(a)):.1f}, first-seat win {np.mean(a[0::2]>0):.3f} second-seat win {np.mean(a[1::2]>0):.3f}")
    pickle.dump(a, open(f"data/eval_{os.path.basename(CKPT)[:-3]}{'_search' if SEARCH else ''}{'_sim' if SIM else ''}{'_vsearch' if VSEARCH else ''}{'_hval' if HVAL else ''}{TAGE}_{NG}.pkl", "wb"))
