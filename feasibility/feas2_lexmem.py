"""
Step 2 (proper): lexicon memory layer on real self-play boards, full ENABLE lexicon, d=256.
Label = reachable-within-<=3-missing-letters, weighted by P(draw completes) * score (exact).
Train: listwise softmax over the rules-masked candidate set (target = 0.5*weight-normalized + 0.5*uniform over reachable).
Eval (test games, never seen):
  NDCG@32 (gains = weight)          vs GADDAG-style baseline (playable-now words by score, rest random)
  percentile of reachable words by #missing letters (0..3)  [random = 0.50; lower = better]
  same for HELD-OUT words (5% of lexicon never a positive in training)
  recall@32 of the top-32-by-weight words
  all conditioned on mask size.
Env: STEPS, B, TAG, SLOT(1/0), QL(query layers), D, LR
"""
import os, sys, time, pickle, math, numpy as np, torch, torch.nn.functional as F, random
from torch.utils.data import DataLoader
from lexmem.core import *; from lexmem.data import *; from lexmem.model import *
import warnings; warnings.filterwarnings("ignore")
STEPS = int(os.environ.get("STEPS", 4000)); B = int(os.environ.get("B", 48)); TAG = os.environ.get("TAG", "run")
SLOT = os.environ.get("SLOT", "1") == "1"; QL = int(os.environ.get("QL", 3)); D = int(os.environ.get("D", 256))
LR = float(os.environ.get("LR", 3e-4)); NW_ = int(os.environ.get("NWORK", os.environ.get("SLURM_CPUS_PER_TASK", 12))); ALPHA = float(os.environ.get("ALPHA", 0.5))
torch.manual_seed(0); np.random.seed(0); random.seed(0)
dev = "cuda"
lex = Lexicon(os.environ.get("LEXICON", os.path.join(ROOT, "enable1.txt"))); heldout = holdout_mask(lex)
games_tr = pickle.load(open(f"data/games_{os.environ.get('TRAIN','train')}.pkl", "rb")); games_te = pickle.load(open(f"data/games_{os.environ.get('TEST','test')}.pkl", "rb"))
NEV = int(os.environ.get("NEV", 400))
print(f"train games {len(games_tr)}  test games {len(games_te)}  lexicon {lex.NW}  heldout {heldout.sum()}", flush=True)

# ---------------- fixed eval set (full candidate sets, no subsampling)
def build_eval(games, n, p_construct, miss_choices, seed):
    ds = LaneDataset(lex, games, heldout, n_neg=10**6, n_pos=10**6, p_construct=p_construct,
                     exclude_heldout=False, seed=seed, miss_choices=miss_choices)
    ds.rng = random.Random(seed)
    return [ds[i] for i in range(n)]
t = time.time()
EV = {"real": build_eval(games_te, NEV, 0.0, (0,), 1),
      "constructed": build_eval(games_te, NEV, 1.0, (0, 1, 2, 3), 2)}
print(f"eval sets built {time.time()-t:.0f}s; mask sizes real median {np.median([e['masksize'] for e in EV['real']]):.0f} "
      f"constructed median {np.median([e['masksize'] for e in EV['constructed']]):.0f}", flush=True)

def ndcg(order_gain, ideal_gain, k=32):
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (order_gain[:k] * disc[:len(order_gain[:k])]).sum(); idcg = (ideal_gain[:k] * disc[:len(ideal_gain[:k])]).sum()
    return dcg / idcg if idcg > 0 else np.nan

@torch.no_grad()
def full_keys(model):
    ks = []
    for i in range(0, lex.NW, 8192):
        idx = torch.arange(i, min(i + 8192, lex.NW), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): ks.append(model.wenc(idx).float())
    return torch.cat(ks)

@torch.no_grad()
def evaluate(model, items, keys, tag):
    model.eval()
    res = dict(ndcg=[], ndcg_gaddag=[], ndcg_score=[], rec32=[], msz=[])
    pct = {m: [] for m in range(4)}; pct_ho = {m: [] for m in range(4)}; top1_ok = []
    for it in items:
        b = collate([it]); b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(b, keys=keys)[0].float().cpu().numpy()
        n = len(it["cand"]); lg = lg[:n]
        w = it["weight"]; ok = it["ok"]; miss = it["miss"]; ho = heldout[it["cand"]]
        order = np.argsort(-lg, kind="stable")
        ideal = np.sort(w)[::-1]
        res["ndcg"].append(ndcg(w[order], ideal)); res["msz"].append(it["masksize"])
        # GADDAG baseline: playable-now by score, others random
        gsc = np.where(miss == 0, it["score"], 0) + np.random.rand(n) * 1e-3
        res["ndcg_gaddag"].append(ndcg(w[np.argsort(-gsc)], ideal))
        res["ndcg_score"].append(ndcg(w[np.argsort(-it["score"] - np.random.rand(n) * 1e-3)], ideal))
        top = set(order[:32].tolist()); topw = set(np.argsort(-w)[:32].tolist()); topw = {i for i in topw if w[i] > 0}
        if topw: res["rec32"].append(len(top & topw) / len(topw))
        top1_ok.append(bool(ok[order[0]]))
        rank = np.empty(n); rank[order] = np.arange(n)
        pctl = rank / max(n - 1, 1)
        for m in range(4):
            sel = ok & (miss == m)
            if sel.any():
                pct[m].append(pctl[sel].mean())
                if (sel & ho).any(): pct_ho[m].append(pctl[sel & ho].mean())
    model.train()
    out = {k: float(np.nanmean(v)) for k, v in res.items() if k != "msz"}
    out.update({f"pct{m}": float(np.mean(pct[m])) if pct[m] else np.nan for m in range(4)})
    out.update({f"pctHO{m}": float(np.mean(pct_ho[m])) if pct_ho[m] else np.nan for m in range(4)})
    out["top1ok"] = float(np.mean(top1_ok))
    # conditioned on mask size
    msz = np.array(res["msz"]); nd = np.array(res["ndcg"])
    for lo, hi in ((2, 50), (50, 500), (500, 5000), (5000, 10**6)):
        s = (msz >= lo) & (msz < hi)
        out[f"ndcg[{lo},{hi})"] = float(np.nanmean(nd[s])) if s.any() else np.nan
    print(f"  [{tag}] " + " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out

def loss_fn(logits, batch):
    valid = batch["valid"]; w = batch["weight"] * valid; ok = batch["ok"].float() * valid
    tw = w / w.sum(1, keepdim=True).clamp_min(1e-9); tu = ok / ok.sum(1, keepdim=True).clamp_min(1e-9)
    target = ALPHA * tw + (1 - ALPHA) * tu
    lp = F.log_softmax(logits.float(), 1)
    return -(target * lp).sum(1).mean()

model = LexMem(lex, D=D, slot=SLOT, qlayers=QL).to(dev)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  slot={SLOT} QL={QL} D={D}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
ds = LaneDataset(lex, games_tr, heldout, n_neg=int(os.environ.get("NNEG",384)), n_pos=int(os.environ.get("NPOS",192)), p_construct=0.5, seed=0)
dl = DataLoader(ds, batch_size=B, num_workers=NW_, collate_fn=collate, persistent_workers=True, prefetch_factor=4)

keys = full_keys(model)
print("untrained:"); evaluate(model, EV["real"][:150], keys, "real"); evaluate(model, EV["constructed"][:150], keys, "constructed")
t0 = time.time(); step = 0; la = 0.0
for batch in dl:
    batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(batch)
    loss = loss_fn(logits, batch)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = 0.98 * la + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.3f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    if step % int(os.environ.get("EVAL_EVERY", 1000)) == 0 or step == STEPS:
        keys = full_keys(model)
        evaluate(model, EV["real"][:150], keys, "real"); evaluate(model, EV["constructed"][:150], keys, "constructed")
        torch.save(model.state_dict(), f"data/lexmem_{TAG}.pt")
    if step >= STEPS: break
print("FINAL", flush=True)
keys = full_keys(model)
R = evaluate(model, EV["real"], keys, "real"); C = evaluate(model, EV["constructed"], keys, "constructed")
pickle.dump(dict(real=R, constructed=C, cfg=dict(STEPS=STEPS, B=B, SLOT=SLOT, QL=QL, D=D, LR=LR, ALPHA=ALPHA)), open(f"data/result_{TAG}.pkl", "wb"))
