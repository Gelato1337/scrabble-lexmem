"""Rack inference with the lexicon memory (round 6): after the opponent (HastyBot) plays, which tiles did it keep?
Data: data/infer_train.npz, data/infer_test.npz (prep/infer_positions.py; held-out = the test games).
Input, from the observer's view before the opponent's move: the board; the placed tiles P (in the planning net's "leave" slot);
U - P, the tiles that could be the rest of the opponent's rack or the bag ("unseen" slot); globals; lane units as in the
planning net (train/train_lexvalue.py): the NLANES - 1 best lanes of the board before the move, two sides each (side 0 = the
opponent with P + up to 3 unseen letters, side 1 = unseen letters only; ARM=lex retrieves over the side's rules mask in the
full lexicon, ARM=nolex has the query alone), plus one unit for the played move (its lane on the board after the move and its
score; no retrieval). Output: for each of the 27 tiles a distribution over the kept count 0 / 1 / 2 / 3+ (cross-entropy),
as a correction to the baseline: logits = baseline log-probabilities + the net's 108 outputs, the last layer starting at zero
(an untrained net predicts the baseline; what it learns is inference, not the arithmetic of the prior).
Baseline (no inference): the exact posterior of the kept tiles given only that the rack held P, the rack being a uniform
7-tile draw from U: P(leave) ∝ Π_x C(U_x, P_x + leave_x); per-tile marginals by dynamic programming.
Metric (NEV_POS held-out positions): log-loss of the true kept counts summed over the 27 tiles, in bits, model vs baseline;
bits saved per position, by phase (bag before the move). The final evaluation also writes data/<TAG>_evbits.npy.
Arms as the planning benchmark: QINIT=fut_train (pretrained query encoder, trained, both arms), NO_XCHK=1 (set in the env).
Env: ARM QINIT MAX_TRAIN STEPS B LR SEED NWORK NEV_POS EVAL_EVERY TAG
Usage: ARM=lex QINIT=fut_train NO_XCHK=1 MAX_TRAIN=3000 STEPS=4000 SEED=0 TAG=lexinfer_lex_3000_s0 python train/train_lexinfer.py"""
import os
os.environ["NOUT"] = "108"   # 27 tiles x 4 count classes; read by train_lexvalue at import
import sys, time, math, random, importlib.util, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
spec = importlib.util.spec_from_file_location("tlv", os.path.join(os.path.dirname(__file__), "train_lexvalue.py")); tlv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tlv)
from lexmem.core import *
from lexmem.data import lane_tokens
from lexmem.replay import place
from lexmem.infer import baseline_logp
E = os.environ.get; dev = tlv.dev; lex = tlv.lex
MAX_TRAIN = int(E("MAX_TRAIN", 0)); STEPS = int(E("STEPS", 20000)); B = int(E("B", 64)); LR = float(E("LR", 2e-4)); SEED = int(E("SEED", 0))
NW = int(E("NWORK", 28)); NEVP = int(E("NEV_POS", 3000)); TAG = E("TAG", f"lexinfer_{tlv.ARM}"); QINIT = E("QINIT", "fut_train"); NLU = tlv.NL - 1
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
TR = dict(np.load("data/infer_train.npz")); TE = dict(np.load("data/infer_test.npz"))
train_idx = list(range(len(TR["gi"])))
if MAX_TRAIN and MAX_TRAIN < len(train_idx): train_idx = random.Random(123).sample(train_idx, MAX_TRAIN)   # same subsample for every arm / seed
ev_idx = random.Random(7).sample(range(len(TE["gi"])), min(3000, len(TE["gi"])))[:NEVP]   # a prefix of one fixed sample (eval/infer_macondo.py uses it too)
print(f"ARM={tlv.ARM} seed={SEED}: {len(train_idx)} training positions (of {len(TR['gi'])}), {len(ev_idx)} held-out", flush=True)

def units_for(board, blanks, Pi, Upi, bagn, sp):
    cc = cross_checks(lex, board, blanks); U = []
    for (r, c, d, L) in tlv.pick_lanes(board)[:NLU]:
        info = lane_info(board, blanks, cc, r, c, d, L); lt = lane_tokens(info, L)
        if tlv.NO_XCHK: lt[:L, 27:53] = 1.0
        for side in (0, 1):
            u = dict(lane=lt, L=L, side=side, glob=np.array([bagn / 100, sp / 100 * (1 if side == 0 else -1), L / 15, 0.0, float(side)], dtype=np.float32))
            if tlv.LEXARM:
                g, loc = lane_candidates(lex, info, L)
                if len(loc): ok = tlv.side_mask(info, L, loc, Pi, Upi, side); g, loc = g[ok], loc[ok]
                sc = lane_scores(lex, info, L, loc).astype(np.float32) if len(loc) else np.zeros(0, np.float32)
                if len(g) > tlv.CAP: keep = np.argsort(-sc)[:tlv.CAP]; g, sc = g[keep], sc[keep]
                u.update(cand=g.astype(np.int64), score=sc, msize=np.float32(math.log1p(len(loc))), aux=-1)
            U.append(u)
    return U

def item(A, i):
    board = A["board"][i].astype(np.int64); blanks = A["blanks"][i].astype(bool); Pi = A["P"][i].astype(np.int64); Ui = A["U"][i].astype(np.int64)
    r0, c0, d0, L0, w0, s = (int(x) for x in A["move"][i]); letters = str(A["letters"][i]); bagn = int(A["bag"][i])
    sp = int(A["scores"][i][0]) - int(A["scores"][i][1]); n = int(A["leave"][i].sum())
    units = units_for(board, blanks, Pi, Ui - Pi, bagn, sp)
    b2, bl2, _ = place(board, blanks, r0, c0, d0, letters)   # the played move: its lane on the board after the move
    info = lane_info(b2, bl2, cross_checks(lex, b2, bl2), r0, c0, d0, L0); lt = lane_tokens(info, L0)
    if tlv.NO_XCHK: lt[:L0, 27:53] = 1.0
    u = dict(lane=lt, L=L0, side=2, glob=np.array([bagn / 100, sp / 100, L0 / 15, s / 50, 2.0], dtype=np.float32))
    if tlv.LEXARM: u.update(cand=np.zeros(0, np.int64), score=np.zeros(0, np.float32), msize=np.float32(0.0), aux=-1)
    units.append(u)
    return dict(board=board + 1, leave=Pi.astype(np.float32), unseen=(Ui - Pi).astype(np.float32),
                glob=np.array([bagn / 100, sp / 100, int(A["turn"][i]) / 30], dtype=np.float32), units=units, G=np.float32(0), W=np.float32(0), bag=bagn,
                Y=np.minimum(A["leave"][i], 3).astype(np.int64), BL=baseline_logp(Ui, Pi, n).astype(np.float32))
def coll(items):
    b = tlv.collate(items); b["Y"] = torch.tensor(np.stack([it["Y"] for it in items])); b["BL"] = torch.tensor(np.stack([it["BL"] for it in items])); return b
class DS(Dataset):
    def __init__(self, seed): self.rng = random.Random(seed)
    def __len__(self): return 10 ** 9
    def __getitem__(self, k): return item(TR, train_idx[self.rng.randrange(len(train_idx))])
class DSfix(Dataset):
    def __len__(self): return len(ev_idx)
    def __getitem__(self, k): return item(TE, ev_idx[k])

model = tlv.LexValue().to(dev)
sd = torch.load(E("INIT_FUT", "data/future_R1.pt"), map_location=dev)
assert QINIT in ("fut", "fut_train", "random")
if QINIT != "random": model.qenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("qenc.")})
if QINIT == "fut":
    for p in model.qenc.parameters(): p.requires_grad_(False)
if tlv.LEXARM:
    model.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.") and not k.startswith("wenc.slot") and k not in ("wenc.POS", "wenc.LEN")}, strict=False)
    model.refresh_keys()
    for p in model.wenc.parameters(): p.requires_grad_(False)
with torch.no_grad(): model.val[2].weight.zero_(); model.val[2].bias.zero_()   # start at the baseline
params = [p for p in model.parameters() if p.requires_grad]
print(f"query encoder {QINIT}, no cross-checks {tlv.NO_XCHK}; trainable params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
EV = [x for ch in DataLoader(DSfix(), batch_size=256, num_workers=NW, collate_fn=lambda b: b) for x in ch]
LN2 = math.log(2)

@torch.no_grad()
def evaluate(tag):
    model.eval(); rows = []
    for k in range(0, len(EV), 128):
        b = {kk: v.to(dev) for kk, v in coll(EV[k:k + 128]).items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
        lp = torch.log_softmax(b["BL"] + v.float().view(-1, 27, 4), -1); Y = b["Y"][..., None]
        nm = -lp.gather(-1, Y).squeeze(-1).sum(1); nb = -b["BL"].gather(-1, Y).squeeze(-1).sum(1)
        rows += list(zip((nm / LN2).tolist(), (nb / LN2).tolist(), b["bag"].tolist()))
    model.train(); R = np.array(rows)
    if tag == "final": np.save(f"data/{TAG}_evbits.npy", R)   # per held-out position (ev order): model bits, baseline bits, bag
    line = f"[{tag}] held-out n={len(R)}: model {R[:,0].mean():.3f} bits, baseline {R[:,1].mean():.3f} bits, saved {(R[:,1]-R[:,0]).mean():+.3f} ± {(R[:,1]-R[:,0]).std()/np.sqrt(len(R)):.3f}"
    for lo, hi, nm_ in ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (1, 7, "pre-end")):
        k = (R[:, 2] >= lo) & (R[:, 2] <= hi)
        if k.sum(): line += f" | {nm_} (n={int(k.sum())}): saved {(R[k,1]-R[k,0]).mean():+.3f}"
    print(line, flush=True)

evaluate("init")
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
t0 = time.time(); step = 0; la = None
for b in DataLoader(DS(SEED), batch_size=B, num_workers=NW, collate_fn=coll, persistent_workers=True, prefetch_factor=4):
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, _ = model(b)
    loss = F.cross_entropy((b["BL"] + v.float().view(-1, 27, 4)).view(-1, 4), b["Y"].view(-1), reduction="sum") / len(b["Y"])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 500 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la / LN2:.3f} bits", flush=True)
    if step % int(E("EVAL_EVERY", 4000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/{TAG}.pt")
    if step >= STEPS: break
evaluate("final"); print(f"=== saved data/{TAG}.pt", flush=True)
