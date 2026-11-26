"""Planning benchmark, lex vs no-lex. Data: data/plan_v1.pkl (prep/plan_positions.py + go/lexmem_plan): positions with up to 16
candidate plays, each labelled by enumeration with E_opp (opponent's expected best reply) and E_my (my expected best next play,
leave + refill, same board). Planning value PV = score - E_opp + E_my; the planning answer is the candidate with the highest PV.
Model: the stage-2 afterstate net (train/train_lexvalue.py: lexicon-free lanes, per-lane per-side tokens; ARM=lex retrieves over
the whole lexicon, ARM=nolex does not), query encoder frozen in both arms, two regression outputs: E_opp/50 and E_my/50.
Held-out evaluation (positions of games gi % HOLD == 0): R² of E_opp / E_my (pooled, and within position = what decides moves),
decision accuracy and regret vs the exact PV answer, for all positions and for planning-critical ones (the PV-best move beats the
top-scoring move by >= CRIT points), against the no-planning baseline (always the top-scoring move); by phase.
Eval only: EVAL_CKPTS=a.pt,b.pt (weights from another dictionary load fine; word keys are rebuilt for LEXICON); SENS_REF=<labels of the
same positions under another dictionary> adds a report on the quarter of positions whose labels moved most.
QINIT: fut (default) = query encoder from the future-word model, frozen; fut_train = same start, trained; random = fresh and trained
(a control with no dictionary-trained part). HORIZON=2 (labels from go/lexmem_plan H2=1): three outputs E_opp, E_my2, E_opp2 and the two-turn planning value
PV2 = score - E_opp + E_my2 - E_opp2. HORIZON=3 (H3=1 labels): five outputs, + E_my3, E_opp3; PV3 = PV2 + E_my3 - E_opp3.
HOPS / HOP_LAYERS / HOP_LEX: several lookups in a row (train_lexvalue.py). The last evaluation's per-position rows go to
data/planrows_<TAG>.pkl (eval/plan_pairs.py compares runs position by position).
Usage: ARM=lex MAX_TRAIN=50000 STEPS=20000 SEED=0 TAG=lexplan_lex_50k python train/train_lexplan.py"""
import os, sys, time, math, pickle, random, importlib.util, numpy as np, torch, torch.nn.functional as F
HOR = int(os.environ.get("HORIZON", "1")); H2 = HOR >= 2; H3 = HOR == 3
if H2: os.environ["NOUT"] = "5" if H3 else "3"
from torch.utils.data import Dataset, DataLoader
spec = importlib.util.spec_from_file_location("tlv", os.path.join(os.path.dirname(__file__), "train_lexvalue.py")); tlv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tlv)
from lexmem.core import *
E = os.environ.get; dev = tlv.dev; lex = tlv.lex
HOLD = int(E("HOLDOUT_MOD", 10)); MAX_TRAIN = int(E("MAX_TRAIN", 0)); STEPS = int(E("STEPS", 20000)); B = int(E("B", 64)); LR = float(E("LR", 2e-4))
SEED = int(E("SEED", 0)); NW = int(E("NWORK", 28)); NEVP = int(E("NEV_POS", 3000)); CRIT = float(E("CRIT", 5.0)); TAG = E("TAG", f"lexplan_{tlv.ARM}")
LOSS = E("LOSS", "reg")   # reg: regress E_opp, E_my per candidate; rank: + listwise CE on the planning value over each position's candidates
TAU = float(E("TAU", 1.5)); RANK_W = float(E("RANK_W", 1.0)); BPOS = int(E("BPOS", 4))   # rank mode: positions per batch (x up to 16 candidates)
QINIT = E("QINIT", "fut")
# targets: (name, index in the cands tuple, sign in the planning value)
TGT = [("E_opp", 2, -1), ("E_my2", 6, 1), ("E_opp2", 7, -1)] if H2 else [("E_opp", 2, -1), ("E_my", 3, 1)]
if H3: TGT += [("E_my3", 10, 1), ("E_opp3", 11, -1)]
assert not (H2 and LOSS == "rank")
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
    it["G"], it["W"] = np.float32(eo / 50), np.float32(em / 50); it["pos"], it["cand_j"] = i, j
    if H2: c = data[i]["cands"][j]; it["W"], it["G2"] = np.float32(c[6] / 50), np.float32(c[7] / 50)
    if H3: it["W3"], it["G3"] = np.float32(c[10] / 50), np.float32(c[11] / 50)
    return it
class DS(Dataset):   # one random stream per DataLoader worker (a shared seed made every worker draw the same batches)
    def __init__(self, seed): self.seed, self.rng = seed, None
    def __len__(self): return 10 ** 9
    def __getitem__(self, k):
        if self.rng is None: wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + k)
        while True:
            it = item(*train_items[self.rng.randrange(len(train_items))])
            if it is not None: return it
class DSfix(Dataset):
    def __init__(self, keys): self.keys = keys
    def __len__(self): return len(self.keys)
    def __getitem__(self, k): return item(*self.keys[k])
class DSpos(Dataset):   # rank mode: one item = all candidates of one training position
    def __init__(self, seed): self.seed, self.rng = seed, None
    def __len__(self): return 10 ** 9
    def __getitem__(self, k):
        if self.rng is None: wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + k)
        while True:
            i = train_pos[self.rng.randrange(len(train_pos))]
            its = [x for x in (item(i, j) for j in range(len(data[i]["cands"]))) if x is not None]
            if len(its) >= 2: return its
def coll_pos(groups):
    items = [it for g in groups for it in g]; b = coll(items)
    b["grp"] = torch.tensor([gi for gi, g in enumerate(groups) for _ in g]); b["sc"] = torch.tensor([float(data[it["pos"]]["cands"][it["cand_j"]][1]) for it in items])
    return b
def coll(items):
    b = tlv.collate(items); b["pos"] = torch.tensor([it["pos"] for it in items]); b["cand_j"] = torch.tensor([it["cand_j"] for it in items])   # not "cand": that is the lex arm's retrieval tensor
    if H2: b["G2"] = torch.tensor([it["G2"] for it in items])
    if H3: b["W3"] = torch.tensor([it["W3"] for it in items]); b["G3"] = torch.tensor([it["G3"] for it in items])
    return b

model = tlv.LexValue().to(dev)
sd = torch.load(E("INIT_FUT", "data/future_R1.pt"), map_location=dev)
assert QINIT in ("fut", "fut_train", "random")
if QINIT != "random": model.qenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("qenc.")})
if QINIT == "fut":
    for p in model.qenc.parameters(): p.requires_grad_(False)
print(f"query encoder: {dict(fut='future-word model, frozen', fut_train='future-word model, trained', random='fresh, trained')[QINIT]}; horizon {HOR}; lookups {tlv.HOPS}" + (f" ({tlv.HOP_LAYERS} layers after each later one; later ones {'retrieve' if tlv.LEXARM and tlv.HOP_LEX else 'do not retrieve'})" if tlv.HOPS > 1 else ""), flush=True)
if tlv.LEXARM:
    model.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.") and not k.startswith("wenc.slot") and k not in ("wenc.POS", "wenc.LEN")}, strict=False)
    model.refresh_keys()
    for p in model.wenc.parameters(): p.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]; print(f"trainable params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
ev_keys = [(i, j) for i in held_pos for j in range(len(data[i]["cands"]))]
EV = [x for ch in DataLoader(DSfix(ev_keys), batch_size=256, num_workers=NW, collate_fn=lambda b: b) for x in ch if x is not None]
print(f"eval candidates {len(EV)}", flush=True)
SENS = set()   # dictionary-sensitive held-out positions: top quarter by label shift vs a reference labelling (same positions, other dictionary)
if E("SENS_REF"):
    ref = {(r["gi"], r["turn"]): r for r in pickle.load(open(E("SENS_REF"), "rb"))}; shift = {}
    for i in held_pos:
        r = data[i]; q = ref.get((r["gi"], r["turn"]))
        if q is None: continue
        mo = lambda x: np.mean([c[2] for c in x["cands"]]); mm = lambda x: np.mean([c[3] for c in x["cands"]])
        shift[i] = abs(mo(r) - mo(q)) + abs(mm(r) - mm(q))
    if shift:
        thr = np.quantile(list(shift.values()), 0.75); SENS = {i for i, v in shift.items() if v >= thr}
        print(f"dictionary-sensitive positions: {len(SENS)} of {len(shift)} matched (label shift >= {thr:.1f} points)", flush=True)

@torch.no_grad()
def evaluate(tag, subset=None, save=None):   # save: file for the per-position rows (paired comparisons: eval/plan_pairs.py)
    model.eval(); pred = {}
    for k in range(0, len(EV), 128):
        b = {kk: v.to(dev) for kk, v in coll(EV[k:k + 128]).items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
        v = v.float().cpu().numpy() * 50
        for t in range(len(v)): pred[(int(b["pos"][t]), int(b["cand_j"][t]))] = v[t]
    model.train()
    ys, ps, ws = [[] for _ in TGT], [[] for _ in TGT], [[] for _ in TGT]; rows = []; ids = []
    for i in held_pos:
        if subset is not None and i not in subset: continue
        cs = data[i]["cands"]; js = [j for j in range(len(cs)) if (i, j) in pred]
        if len(js) < 2: continue
        sc = np.array([cs[j][1] for j in js], dtype=float); pv, pvh = sc.copy(), sc.copy()
        for t, (_, ix, sg) in enumerate(TGT):
            y = np.array([cs[j][ix] for j in js]); q = np.array([pred[(i, j)][t] for j in js])
            ys[t] += list(y); ps[t] += list(q); ws[t] += list(zip(y - y.mean(), q - q.mean())); pv += sg * y; pvh += sg * q
        best = int(np.argmax(pv)); ch = int(np.argmax(pvh)); base = int(np.argmax(sc))
        rows.append((ch == best, pv[best] - pv[ch], base == best, pv[best] - pv[base], pv[best] - pv[base] >= CRIT, data[i]["bag"])); ids.append(i)
    def r2(y, p): y, p = np.array(y), np.array(p); return 1 - ((y - p) ** 2).mean() / max(y.var(), 1e-9)
    def r2w(z): z = np.array(z); return 1 - ((z[:, 0] - z[:, 1]) ** 2).mean() / max((z[:, 0] ** 2).mean(), 1e-9)
    R = np.array(rows, dtype=float); crit = R[:, 4] == 1
    if save: pickle.dump(dict(pos=np.array(ids), rows=R, cols=("correct", "regret", "base_correct", "base_regret", "critical", "bag")), open(save, "wb"))
    line = (f"[{tag}] positions {len(R)}: R2 " + ", ".join(f"{nm} {r2(ys[t], ps[t]):.3f} (within {r2w(ws[t]):.3f})" for t, (nm, _, _) in enumerate(TGT)) + " | "
            f"decision acc {R[:,0].mean():.3f} regret {R[:,1].mean():.2f} pts (top-score baseline acc {R[:,2].mean():.3f} regret {R[:,3].mean():.2f})")
    if crit.sum(): line += f" | critical n={int(crit.sum())}: acc {R[crit,0].mean():.3f} regret {R[crit,1].mean():.2f} (baseline regret {R[crit,3].mean():.2f})"
    for lo, hi, nm in ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (0, 7, "pre-end/end")):
        k = (R[:, 5] >= lo) & (R[:, 5] <= hi)
        if k.sum(): line += f" | {nm}: acc {R[k,0].mean():.3f} regret {R[k,1].mean():.2f}"
    print(line, flush=True)
def load_ckpt(path):
    """weights from another dictionary: skip dictionary-sized buffers (word keys, letter tables) and rebuild the keys here."""
    sd = torch.load(path, map_location=dev); own = model.state_dict()
    skipped = [k for k, v in sd.items() if k not in own or own[k].shape != v.shape]
    model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    if tlv.LEXARM: model.refresh_keys()
    return skipped
if E("EVAL_CKPTS"):   # evaluation only (e.g. zero-shot transfer to another dictionary)
    for path in E("EVAL_CKPTS").split(","):
        sk = load_ckpt(path); print(f"=== {path} (skipped buffers: {sk})", flush=True)
        evaluate(os.path.basename(path), save=f"data/planrows_{os.path.basename(path)[:-3]}.pkl")
        if SENS: evaluate(os.path.basename(path) + " | dictionary-sensitive", SENS)
    sys.exit(0)
if E("INIT_CKPT"): print(f"init from {E('INIT_CKPT')} (skipped buffers: {load_ckpt(E('INIT_CKPT'))})", flush=True)   # few-shot transfer
evaluate("init")
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
t0 = time.time(); step = 0; la = None
loader = DataLoader(DSpos(SEED), batch_size=BPOS, num_workers=NW, collate_fn=coll_pos, persistent_workers=True, prefetch_factor=4) if LOSS == "rank" else \
         DataLoader(DS(SEED), batch_size=B, num_workers=NW, collate_fn=coll, persistent_workers=True, prefetch_factor=4)
for b in loader:
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
    v = v.float(); loss = F.smooth_l1_loss(v[:, 0], b["G"]) + F.smooth_l1_loss(v[:, 1], b["W"])
    if H2: loss = loss + F.smooth_l1_loss(v[:, 2], b["G2"])
    if H3: loss = loss + F.smooth_l1_loss(v[:, 3], b["W3"]) + F.smooth_l1_loss(v[:, 4], b["G3"])
    if LOSS == "rank":   # listwise: predicted PV = score - 50 E_opp + 50 E_my vs exact PV, softmax over each position's candidates
        pv_hat = b["sc"] - 50 * v[:, 0] + 50 * v[:, 1]; pv = b["sc"] - 50 * b["G"] + 50 * b["W"]; lr_ = 0.0
        for g in b["grp"].unique():
            m = b["grp"] == g
            lr_ = lr_ - (torch.softmax(pv[m] / TAU, 0) * torch.log_softmax(pv_hat[m] / TAU, 0)).sum()
        loss = loss + RANK_W * lr_ / len(b["grp"].unique())
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 500 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}", flush=True)
    if step % int(E("EVAL_EVERY", 4000)) == 0 or step == STEPS:
        evaluate(f"step {step}", save=f"data/planrows_{TAG}.pkl" if step == STEPS else None); torch.save(model.state_dict(), f"data/{TAG}.pt")
        if SENS and step == STEPS: evaluate(f"step {step} | dictionary-sensitive", SENS)
    if step >= STEPS: break
print(f"=== saved data/{TAG}.pt", flush=True)
