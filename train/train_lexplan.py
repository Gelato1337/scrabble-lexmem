"""Planning benchmark, lex vs no-lex. Data: data/plan_v1.pkl (prep/plan_positions.py + go/lexmem_plan): positions with up to 16
candidate plays, each labelled by enumeration with E_opp (opponent's expected best reply) and E_my (my expected best next play,
leave + refill, same board). Planning value PV = score - E_opp + E_my; the planning answer is the candidate with the highest PV.
Model: the stage-2 afterstate net (train/train_lexvalue.py: lexicon-free lanes, per-lane per-side tokens; ARM=lex retrieves over
the whole lexicon, ARM=nolex does not), query encoder frozen in both arms, two regression outputs: E_opp/50 and E_my/50.
Held-out evaluation (positions of games gi % HOLD == 0): R² of E_opp / E_my (pooled, and within position = what decides moves),
decision accuracy and regret vs the exact PV answer, for all positions and for planning-critical ones (the PV-best move beats the
top-scoring move by >= CRIT points), against the no-planning baseline (always the top-scoring move); by phase.
Usage: ARM=lex MAX_TRAIN=50000 STEPS=20000 SEED=0 TAG=lexplan_lex_50k python train/train_lexplan.py"""
import os, sys, time, math, pickle, random, importlib.util, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
spec = importlib.util.spec_from_file_location("tlv", os.path.join(os.path.dirname(__file__), "train_lexvalue.py")); tlv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tlv)
from lexmem.core import *
E = os.environ.get; dev = tlv.dev; lex = tlv.lex
HOLD = int(E("HOLDOUT_MOD", 10)); MAX_TRAIN = int(E("MAX_TRAIN", 0)); STEPS = int(E("STEPS", 20000)); B = int(E("B", 64)); LR = float(E("LR", 2e-4))
SEED = int(E("SEED", 0)); NW = int(E("NWORK", 28)); NEVP = int(E("NEV_POS", 3000)); CRIT = float(E("CRIT", 5.0)); TAG = E("TAG", f"lexplan_{tlv.ARM}")
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
data = pickle.load(open(E("DATA", "data/plan_v1.pkl"), "rb"))
train_pos = [i for i, r in enumerate(data) if r["gi"] % HOLD != 0]; held_pos = [i for i, r in enumerate(data) if r["gi"] % HOLD == 0]
if MAX_TRAIN and MAX_TRAIN < len(train_pos): train_pos = random.Random(123).sample(train_pos, MAX_TRAIN)   # same subsample for every arm / seed
held_pos = random.Random(7).sample(held_pos, min(NEVP, len(held_pos)))
train_items = [(i, j) for i in train_pos for j in range(len(data[i]["cands"]))]
print(f"ARM={tlv.ARM} seed={SEED}: {len(train_pos)} train positions ({len(train_items)} candidates), {len(held_pos)} held-out positions", flush=True)

def item(i, j):
    r = data[i]; d, s, eo, em = data[i]["cands"][j][:4]; _, r0, c0, d0, L0, w0, sc, _ = d
    rec = {k: r[k] for k in ("board", "blanks", "rack", "unseen", "scores", "bag", "turn", "mover")}
    rec["move"] = ("play", r0, c0, d0, L0, designate(lex, r["board"], r["rack"], r0, c0, d0, L0, w0), sc)
    it = tlv.build(rec, (0, 0), None)
    if it is None: return None
    it["G"], it["W"] = np.float32(eo / 50), np.float32(em / 50); it["pos"], it["cand"] = i, j
    return it
class DS(Dataset):
    def __init__(self, seed): self.rng = random.Random(seed)
    def __len__(self): return 10 ** 9
    def __getitem__(self, k):
        while True:
            it = item(*train_items[self.rng.randrange(len(train_items))])
            if it is not None: return it
class DSfix(Dataset):
    def __init__(self, keys): self.keys = keys
    def __len__(self): return len(self.keys)
    def __getitem__(self, k): return item(*self.keys[k])
def coll(items):
    b = tlv.collate(items); b["pos"] = torch.tensor([it["pos"] for it in items]); b["cand"] = torch.tensor([it["cand"] for it in items]); return b

model = tlv.LexValue().to(dev)
sd = torch.load(E("INIT_FUT", "data/future_R1.pt"), map_location=dev)
model.qenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("qenc.")})
for p in model.qenc.parameters(): p.requires_grad_(False)
if tlv.LEXARM:
    model.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.") and not k.startswith("wenc.slot") and k not in ("wenc.POS", "wenc.LEN")}, strict=False)
    model.refresh_keys()
    for p in model.wenc.parameters(): p.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]; print(f"trainable params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
ev_keys = [(i, j) for i in held_pos for j in range(len(data[i]["cands"]))]
EV = [x for ch in DataLoader(DSfix(ev_keys), batch_size=256, num_workers=NW, collate_fn=lambda b: b) for x in ch if x is not None]
print(f"eval candidates {len(EV)}", flush=True)

@torch.no_grad()
def evaluate(tag):
    model.eval(); pred = {}
    for k in range(0, len(EV), 128):
        b = {kk: v.to(dev) for kk, v in coll(EV[k:k + 128]).items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
        v = v.float().cpu().numpy() * 50
        for t in range(len(v)): pred[(int(b["pos"][t]), int(b["cand"][t]))] = v[t]
    model.train()
    yo, po, ym, pm, wo, wm = [], [], [], [], [], []; rows = []
    for i in held_pos:
        cs = data[i]["cands"]; js = [j for j in range(len(cs)) if (i, j) in pred]
        if len(js) < 2: continue
        to = np.array([cs[j][2] for j in js]); tm = np.array([cs[j][3] for j in js]); sc = np.array([cs[j][1] for j in js], dtype=float)
        qo = np.array([pred[(i, j)][0] for j in js]); qm = np.array([pred[(i, j)][1] for j in js])
        yo += list(to); po += list(qo); ym += list(tm); pm += list(qm)
        wo += list(zip(to - to.mean(), qo - qo.mean())); wm += list(zip(tm - tm.mean(), qm - qm.mean()))
        pv = sc - to + tm; best = int(np.argmax(pv)); ch = int(np.argmax(sc - qo + qm)); base = int(np.argmax(sc))
        rows.append((ch == best, pv[best] - pv[ch], base == best, pv[best] - pv[base], pv[best] - pv[base] >= CRIT, data[i]["bag"]))
    def r2(y, p): y, p = np.array(y), np.array(p); return 1 - ((y - p) ** 2).mean() / max(y.var(), 1e-9)
    def r2w(z): z = np.array(z); return 1 - ((z[:, 0] - z[:, 1]) ** 2).mean() / max((z[:, 0] ** 2).mean(), 1e-9)
    R = np.array(rows, dtype=float); crit = R[:, 4] == 1
    line = (f"[{tag}] positions {len(R)}: R2 E_opp {r2(yo, po):.3f} (within {r2w(wo):.3f}), E_my {r2(ym, pm):.3f} (within {r2w(wm):.3f}) | "
            f"decision acc {R[:,0].mean():.3f} regret {R[:,1].mean():.2f} pts (top-score baseline acc {R[:,2].mean():.3f} regret {R[:,3].mean():.2f})")
    if crit.sum(): line += f" | critical n={int(crit.sum())}: acc {R[crit,0].mean():.3f} regret {R[crit,1].mean():.2f} (baseline regret {R[crit,3].mean():.2f})"
    for lo, hi, nm in ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (0, 7, "pre-end/end")):
        k = (R[:, 5] >= lo) & (R[:, 5] <= hi)
        if k.sum(): line += f" | {nm}: acc {R[k,0].mean():.3f} regret {R[k,1].mean():.2f}"
    print(line, flush=True)
evaluate("init")
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
t0 = time.time(); step = 0; la = None
for b in DataLoader(DS(SEED), batch_size=B, num_workers=NW, collate_fn=coll, persistent_workers=True, prefetch_factor=4):
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
    v = v.float(); loss = F.smooth_l1_loss(v[:, 0], b["G"]) + F.smooth_l1_loss(v[:, 1], b["W"])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 500 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}", flush=True)
    if step % int(E("EVAL_EVERY", 4000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/{TAG}.pt")
    if step >= STEPS: break
print(f"=== saved data/{TAG}.pt", flush=True)
