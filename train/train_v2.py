"""Track A v2 training (actor-critic, conservative): only game results train it.
Utility of an outcome: U = win (1 / 0.5 / 0) + final spread / 400.
Phase 1  V (value head only; the trunk is not touched before the KL-anchored phase 3): lambda-returns of the future spread change and of the win (as train_td.py; LAM=1 in round 1), position-only pass, VSTEPS.
Phase 2  refresh V, then per player-sequence U(s) = P(win) + (spread now + predicted future change) / 400; terminal = actual U.
         delta_i = U(s_{i+1}) - U(s_i) (my move + their reply), advantage A_i = GAE(delta, LAM_A).
Phase 3  policy, PSTEPS: advantage-weighted likelihood  -w_i log pi(a_i | s_i),  w = min(exp(A / (BETA_A * sd(A))), WMAX), normalised;
         branch pairs (selfplay_v2.py): BCE(sigmoid(l_a - l_b), sigmoid(dU / TAU_B)) * WB  (shared-draw comparisons, low noise);
         KL(pi_start || pi) * BETA_KL keeps each round a small step; V keeps training on its targets.
Usage: TRAIN=v2_w_g1_0,... INIT=ckpt TAG=v2_w_g1 PYTHONPATH=. python train/train_v2.py"""
import os, time, math, copy, pickle, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
LAM = float(E("LAM", 0.7)); LAM_A = float(E("LAM_A", 0.8)); VSTEPS = int(E("VSTEPS", 3000)); PSTEPS = int(E("PSTEPS", 6000)); B = int(E("B", 48))
LR = float(E("LR", 5e-5)); BETA_A = float(E("BETA_A", 1.0)); WMAX = float(E("WMAX", 10)); BETA_KL = float(E("BETA_KL", 1.0))
WB = float(E("WB", 1.0)); TAU_B = float(E("TAU_B", 0.05)); TAG = E("TAG", "v2"); NW = int(E("NWORK", 16))
torch.manual_seed(0); random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex).to(dev)
if E("INIT"): model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); print("init from", E("INIT"), flush=True)
anchor = copy.deepcopy(model).eval()
for p in anchor.parameters(): p.requires_grad_(False)
games = [g for t in E("TRAIN").split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]
flat = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti in range(len(tr))]
HOLD = int(E("HOLDOUT_MOD", 10))   # games with gi % HOLD == 0 are held out of all training: their branch pairs test whether the signal generalises
flat_tr = [(gi, ti) for gi, ti in flat if HOLD <= 0 or gi % HOLD != 0]
print(f"{len(games)} games, {len(flat)} positions, {sum('branch' in games[gi][0][ti] for gi, ti in flat)} branch pairs", flush=True)
helper = MoveDataset(lex, [], exch=True)
def pos_item(rec): return helper.build_pos(rec["board"], rec["rack"], rec["unseen"], rec["scores"], rec["bag"], rec["turn"])
@torch.no_grad()
def refresh():
    model.eval(); out, buf = [], []
    def flush():
        bt = collate_moves(buf); bt = {k: v.to(dev) for k, v in bt.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): _, v = model.qv(bt, with_moves=False)
        v = v.float(); out.append(torch.stack([v[:, 0], torch.sigmoid(v[:, 1])], 1).cpu().numpy()); buf.clear()
    for gi, ti in flat:
        buf.append(pos_item(games[gi][0][ti]))
        if len(buf) == 4096: flush()
    if buf: flush()
    model.train(); V = np.concatenate(out); return {k: V[i] for i, k in enumerate(flat)}
def seqs():
    for gi, (tr, sc) in enumerate(games):
        final = {0: sc[0] - sc[1], 1: sc[1] - sc[0]}
        for p in (0, 1): yield gi, [ti for ti, r in enumerate(tr) if r["mover"] == p], final[p]
def value_targets(vpos, lam):
    T = {}
    for gi, idx, fin in seqs():
        G = W = None; tr = games[gi][0]
        for j in range(len(idx) - 1, -1, -1):
            r = tr[idx[j]]; sp = r["scores"][0] - r["scores"][1]
            if j == len(idx) - 1: G = (fin - sp) / 100; W = 1.0 if fin > 0 else 0.5 if fin == 0 else 0.0
            else:
                rn = tr[idx[j + 1]]; vb = vpos[(gi, idx[j + 1])]
                G = ((rn["scores"][0] - rn["scores"][1]) - sp) / 100 + (1 - lam) * vb[0] + lam * G; W = (1 - lam) * vb[1] + lam * W
            T[(gi, idx[j])] = (np.float32(G), np.float32(W))
    return T
def advantages(vpos):
    A = {}
    for gi, idx, fin in seqs():
        tr = games[gi][0]; U = []
        for ti in idx:
            r = tr[ti]; v = vpos[(gi, ti)]; U.append(v[1] + ((r["scores"][0] - r["scores"][1]) + 100 * v[0]) / 400)
        U.append((1.0 if fin > 0 else 0.5 if fin == 0 else 0.0) + fin / 400)
        a = 0.0
        for j in range(len(idx) - 1, -1, -1): a = (U[j + 1] - U[j]) + LAM_A * a; A[(gi, idx[j])] = a
    return A
def vloss(v, G, W): return F.smooth_l1_loss(v[:, 0], G) + F.binary_cross_entropy_with_logits(v[:, 1], W)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
# ---- phase 1: V
t0 = time.time(); vp = refresh(); T = value_targets(vp, float(E("LAM1", LAM))); print(f"phase 1 targets in {time.time()-t0:.0f}s", flush=True)
class DV(Dataset):
    def __init__(self, s): self.rng = random.Random(s)
    def __len__(self): return 10 ** 8
    def __getitem__(self, i):
        gi, ti = flat_tr[self.rng.randrange(len(flat_tr))]; it = pos_item(games[gi][0][ti]); it["G"], it["W"] = T[(gi, ti)]; return it
def cv(batch):
    b = collate_moves(batch); b["G"] = torch.tensor([x["G"] for x in batch]); b["W"] = torch.tensor([x["W"] for x in batch]); return b
opt_v = torch.optim.AdamW(model.vh.parameters(), lr=float(E("LR_V", 3e-4)), weight_decay=0.01)   # phase 1 touches only the value head
step = 0; la = None
for b in DataLoader(DV(1), batch_size=256, num_workers=NW, collate_fn=cv):
    b = {k: v.to(dev) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): _, v = model.qv(b, with_moves=False)
    loss = vloss(v.float(), b["G"], b["W"]); opt_v.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True); loss.backward(); opt_v.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 500 == 0: print(f"V step {step} loss {la:.4f} |V-G| {(v.float()[:,0]-b['G']).abs().mean().item()*100:.1f} pts", flush=True)
    if step >= VSTEPS: break
# ---- phase 2: advantages
vp = refresh(); T = value_targets(vp, LAM); A = advantages(vp)
items = [(gi, ti) for gi, ti in flat_tr if games[gi][0][ti]["move"][0] in ("play", "exchange")]
av = np.array([A[k] for k in items]); sdA = av.std() + 1e-8
br = np.array([games[gi][0][ti]["branch"][1] for gi, ti in items if "branch" in games[gi][0][ti]])
print(f"advantages: mean {av.mean():+.4f} sd {sdA:.4f} (U units; 0.01 = 1 win-% or 4 pts); "
      f"branch dU: n={len(br)} mean {br.mean() if len(br) else 0:+.4f} sd {br.std() if len(br) else 0:.4f}", flush=True)
def same(m, b): return (m[3] == 2 and b[3] == 2 and tuple(m[5]) == tuple(b[5])) or (m[3] != 2 and b[3] != 2 and tuple(m[1:6]) == tuple(b[1:6]))
class DP(Dataset):
    def __init__(self, s): self.rng = random.Random(s)
    def __len__(self): return 10 ** 8
    def __getitem__(self, i):
        while True:
            gi, ti = items[self.rng.randrange(len(items))]; r = games[gi][0][ti]; m = r["move"]
            it = helper.build(r, 0.0, target_keep=np.array(m[1])) if m[0] == "exchange" else helper.build(r, 0.0, target_rcd=(m[1], m[2], m[3], m[5]))
            if it is None: continue
            it["w"] = np.float32(min(math.exp(A[(gi, ti)] / (BETA_A * sdA)), WMAX)); it["bidx"], it["dU"] = -1, np.float32(0)
            if "branch" in r:
                bm, dU = r["branch"]; j = next((j for j, x in enumerate(it["_moves"]) if same(x, bm)), -1)
                if j >= 0: it["bidx"], it["dU"] = j, np.float32(dU)
            p = pos_item(r); it["G"], it["W"] = T[(gi, ti)]; it.update({f"p_{k}": v for k, v in p.items()}); return it
def cp(batch):
    b = collate_moves(batch); bp = collate_moves([{k[2:]: v for k, v in x.items() if k.startswith("p_")} for x in batch])
    for k in ("w", "bidx", "dU", "G", "W"): b[k] = torch.tensor([x[k] for x in batch])
    return b, bp
step = 0; la = None; t0 = time.time()
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, PSTEPS) / PSTEPS)))
for b, bp in DataLoader(DP(2), batch_size=B, num_workers=NW, collate_fn=cp, persistent_workers=True, prefetch_factor=4):
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}; bp = {k: v.to(dev, non_blocking=True) for k, v in bp.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        pl, _ = model(b); _, v = model.qv(bp, with_moves=False)
        with torch.no_grad(): pa, _ = anchor(b)
    pl, pa, v = pl.float(), pa.float(), v.float(); logp = torch.log_softmax(pl, 1); n = len(b["target"]); ar = torch.arange(n, device=dev)
    w = b["w"] / b["w"].mean(); l_awr = -(w * logp[ar, b["target"]]).mean()
    lpa = torch.log_softmax(pa, 1); l_kl = (lpa.exp() * (lpa - logp)).sum(1).mean()
    hb = b["bidx"] >= 0; l_br = torch.zeros((), device=dev)
    if hb.any():
        d = pl[ar[hb], b["target"][hb]] - pl[ar[hb], b["bidx"][hb]]
        l_br = F.binary_cross_entropy_with_logits(d, torch.sigmoid(b["dU"][hb] / TAU_B))
    loss = l_awr + BETA_KL * l_kl + WB * l_br + vloss(v, b["G"], b["W"])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 200 == 0: print(f"P step {step} t={time.time()-t0:.0f}s loss {la:.4f}  awr {l_awr.item():.3f} kl {l_kl.item():.4f} branch {l_br.item():.3f} ({int(hb.sum())}/batch)", flush=True)
    if step >= PSTEPS: break
@torch.no_grad()
def branch_check(net, tag):
    """held-out branch pairs: does the net's logit difference l_a - l_b predict which move won its shared-draw playout?"""
    net.eval(); rows = []
    for gi, ti in flat:
        if HOLD <= 0 or gi % HOLD != 0: continue
        r = games[gi][0][ti]
        if "branch" not in r or r["move"][0] not in ("play", "exchange"): continue
        m = r["move"]; it = helper.build(r, 0.0, target_keep=np.array(m[1])) if m[0] == "exchange" else helper.build(r, 0.0, target_rcd=(m[1], m[2], m[3], m[5]))
        if it is None: continue
        bm, dU = r["branch"]; j = next((j for j, x in enumerate(it["_moves"]) if same(x, bm)), -1)
        if j < 0: continue
        bt = {k: v.to(dev) for k, v in collate_moves([it]).items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, _ = net(bt)
        pl = pl[0].float(); u = int(r["unseen"].sum())
        rows.append((float(pl[it["target"]] - pl[j]), dU, "pre-endgame/endgame" if u <= 14 else "late (8-30)" if r["bag"] <= 30 else "early/middle (31+)"))
    if not rows: print(f"[{tag}] no held-out branch pairs"); return
    d = np.array([x[0] for x in rows]); u = np.array([x[1] for x in rows]); nz = u != 0
    print(f"[{tag}] held-out branch pairs n={len(rows)}: sign agreement {np.mean(np.sign(d[nz]) == np.sign(u[nz])):.3f} (0.5 = no signal), "
          f"corr(l_a - l_b, dU) {np.corrcoef(d, u)[0,1]:+.3f}", flush=True)
    for ph in sorted(set(x[2] for x in rows)):
        k = np.array([x[2] == ph for x in rows]) & nz
        if k.sum() > 30: print(f"    {ph:22s} n={k.sum():6d} sign agreement {np.mean(np.sign(d[k]) == np.sign(u[k])):.3f} corr {np.corrcoef(d[k], u[k])[0,1]:+.3f}", flush=True)
    net.train()
branch_check(anchor, "start net"); branch_check(model, "trained net")
torch.save(model.state_dict(), f"data/policy_{TAG}.pt"); print(f"=== saved data/policy_{TAG}.pt", flush=True)
