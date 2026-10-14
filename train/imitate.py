"""Imitation + value training on HastyBot games, with/without the lexicon branch. Then play vs HastyBot.
Env: TRAIN, TEST (games tags), LEXICON, STEPS, B, USE_LEX (1/0), RETR (retrieval ckpt), TAG, NWORK, PLAY (games vs HastyBot at the end)"""
import os, sys, time, math, pickle, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from lexmem.core import *; from lexmem.policy import *
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get
STEPS = int(E("STEPS", 3000)); B = int(E("B", 32)); USE_LEX = E("USE_LEX", "0") == "1"; TAG = E("TAG", "imit"); LR = float(E("LR", 3e-4))
NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); PG = E("PG", "0") == "1"; EI = E("EI", "0") == "1"; TAU = float(E("TAU", 2.0)); MARGIN = float(E("MARGIN", 3.0)); BETA = float(E("BETA", 0.5)); PG_TEMP = float(E("TEMP", 0.7)); PPO_EPS = float(E("PPO_EPS", 0.2)); PLAY = int(E("PLAY", 0)); NEV = int(E("NEV", 300)); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))
games_tr = [g for t in E('TRAIN', 'hasty_train').split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]   # TRAIN=a,b concatenates
games_te = pickle.load(open(f"data/games_{E('TEST','hasty_test')}.pkl", "rb"))
print(f"train games {len(games_tr)} test games {len(games_te)} use_lex={USE_LEX}", flush=True)
model = PolicyNet(lex, use_lex=USE_LEX).to(dev)
if USE_LEX:
    if E("RETR"): model.load_retrieval(E("RETR")); print("retrieval init from", E("RETR"))
    else: model.refresh_keys()
if E("INIT"): model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); print("policy init from", E("INIT"), flush=True)
anchor = None
if EI and BETA > 0:
    import copy; anchor = copy.deepcopy(model).eval()
    for p in anchor.parameters(): p.requires_grad_(False)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
KPLY = int(E("KPLY", 4)); EXCH = E("EXCH", "0") == "1"   # EXCH=1: exchanges are move tokens too
ds = MoveDataset(lex, games_tr, seed=0, use_lex=USE_LEX, kply=KPLY, tau=TAU, exch=EXCH, ei=EI)
dl = DataLoader(ds, batch_size=B, num_workers=NW_, collate_fn=collate_moves, persistent_workers=NW_ > 0, prefetch_factor=4 if NW_ > 0 else None)
ev = MoveDataset(lex, games_te, seed=1, use_lex=USE_LEX, kply=KPLY, exch=EXCH, ei=EI); ev.rng = random.Random(1); EV = [ev[i] for i in range(NEV)]
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))

@torch.no_grad()
def evaluate(tag):
    model.eval(); acc = []; top3 = []; vmae = []; v0mae = []; vkbase = []
    for i in range(0, len(EV), 16):
        b = collate_moves(EV[i:i + 16]); b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, v = model(b); _, v0 = model(b, with_moves=False)
        pl = pl.float(); order = pl.argsort(1, descending=True)
        acc += (order[:, 0] == b["target"]).tolist(); top3 += (order[:, :3] == b["target"][:, None]).any(1).tolist(); vmae += (v.float() - b["value"]).abs().tolist(); v0mae += (v0.float() - b["value_k"]).abs().tolist(); vkbase += b["value_k"].abs().tolist()
    model.train(); print(f"  [{tag}] hasty-move top1={np.mean(acc):.3f} top3={np.mean(top3):.3f} value MAE={np.mean(vmae)*100:.1f} pts | {KPLY}-ply spread MAE={np.mean(v0mae)*100:.1f} pts (predict-zero baseline {np.mean(vkbase)*100:.1f})", flush=True)
    return np.mean(acc), np.mean(vmae)

evaluate("untrained"); t0 = time.time(); step = 0; la = None
for b in dl:
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, v = model(b); _, v0 = model(b, with_moves=False)
    if EI:   # expert iteration: soft target from the sim, weighted by the sim's top-2 margin; KL anchor to the initial policy
        logp = torch.log_softmax(pl.float(), 1); wm = (b["margin"] >= MARGIN).float()
        if E("CW", "0") == "1": wm = wm * b["cw"]   # cost weighting: positions where the candidates differ more count more
        ce = -(b["soft"] * logp).sum(1); loss = (ce * wm).sum() / wm.sum().clamp_min(1)
        if anchor is not None:
            with torch.no_grad():
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pa, _ = anchor(b)
            pa = torch.log_softmax(pa.float(), 1); loss = loss + BETA * (pa.exp() * (pa - logp)).sum(1).mean()
    elif PG:   # PPO-clip on the behaviour policy softmax(logits/TEMP); advantages normalised per batch
        logp = torch.log_softmax(pl.float() / PG_TEMP, 1).gather(1, b["target"][:, None]).squeeze(1)
        adv = b["adv"]; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        ratio = torch.exp(logp - b["logp_old"]); pol_loss = -torch.min(ratio * adv, ratio.clamp(1 - PPO_EPS, 1 + PPO_EPS) * adv).mean()
        ent = -(torch.softmax(pl.float() / PG_TEMP, 1) * torch.log_softmax(pl.float() / PG_TEMP, 1)).sum(1).mean()
        loss = pol_loss - 0.01 * ent
    else:
        ce = F.cross_entropy(pl.float(), b["target"], reduction="none"); wp = b["w_pol"]
        loss = (ce * wp).sum() / (wp.sum().clamp_min(1))
    loss = loss + 0.5 * F.smooth_l1_loss(v.float(), b["value"]) + 1.0 * F.smooth_l1_loss(v0.float(), b["value_k"])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.3f}", flush=True)
    if step % int(E("EVAL_EVERY", 1000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/policy_{TAG}.pt")
    if step >= STEPS: break
acc, vmae = evaluate("final")
if PLAY:
    from lexmem.harness import HastyBot, play_vs_hasty
    hasty = HastyBot(); model.eval(); helper = MoveDataset(lex, games_te, use_lex=USE_LEX)
    @torch.no_grad()
    def net_policy(lex_, board, blanks, cc, rack, unseen, rng):
        rec = dict(board=board, blanks=blanks, rack=rack, unseen=unseen, bag=max(0, int(unseen.sum()) - 7), scores=(0, 0), turn=0)
        it = helper.build(rec, 0.0)
        if it is None: return None
        b = collate_moves([it]); b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, v = model(b)
        k = int(pl[0].float().argmax()); mvs = legal_moves(lex_, board, blanks, cc, rack); mvs.sort(key=lambda m: -m[0])
        # rebuild the same ordering as build(): top MAXM by score
        return mvs[:MAXM][k]
    res = []
    for g in range(PLAY):
        spread, scores, log = play_vs_hasty(lex, net_policy, hasty, 5000 + g, we_first=(g % 2 == 0)); res.append(spread)
        if g % 10 == 9: print(f"  played {g+1}: win {np.mean(np.array(res) > 0):.3f} spread {np.mean(res):+.1f}", flush=True)
    res = np.array(res); print(f"NET({TAG}) vs HastyBot: {PLAY} games, win rate {np.mean(res > 0):.3f}, mean spread {res.mean():+.1f} ± {res.std()/np.sqrt(len(res)):.1f}", flush=True); hasty.close()
    pickle.dump(dict(acc=acc, vmae=vmae, spreads=res, use_lex=USE_LEX), open(f"data/result_{TAG}.pkl", "wb"))
