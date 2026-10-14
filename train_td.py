"""Track A training: TD(lambda) on self-play games. The only reward is the game result (final spread and win/loss);
no teacher, no search.
1. Refresh: the current net's position value V(s) for every stored position, and for its K alternative racks (selfplay_td.py).
2. Targets per player-sequence s_0, s_1, ... (one player's own turns), lambda-return of the future spread change and of the win:
     G_i = d_i + (1 - lam) * Vbar(s_{i+1}) + lam * G_{i+1},   G_last = d_last (= final spread - spread now)
     W_i = (1 - lam) * Wbar(s_{i+1}) + lam * W_{i+1},         W_last = 1 / 0.5 / 0
   d_i = change of my spread from my turn i to my turn i+1 (my move + their reply). Vbar = mean over the alternative racks
   (LUCK=1): the value of the position before my draw is known, so my draw luck does not leak into the target.
   Draw luck at s_{i+1} = V(actual rack) - Vbar; reported per game as a variance-reduction diagnostic.
3. Train Q(s, a_chosen) -> (G, W) on move tokens (with moves) and V(s) -> (G, W) on the position-only pass.
Usage: TRAIN=td_x_g1_0,... INIT=ckpt TAG=td_x_g1 STEPS=8000 python train_td.py"""
import os, time, math, pickle, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
LAM = float(E("LAM", 0.7)); LUCK = E("LUCK", "1") == "1"; STEPS = int(E("STEPS", 8000)); B = int(E("B", 48)); LR = float(E("LR", 1e-4)); TAG = E("TAG", "td")
torch.manual_seed(0); random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex).to(dev)
if E("INIT"): model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); print("init from", E("INIT"), flush=True)
games = [g for t in E("TRAIN").split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]
print(f"{len(games)} games, {sum(len(g[0]) for g in games)} positions, lambda={LAM} luck={LUCK}", flush=True)
helper = MoveDataset(lex, [], exch=True)

# ---- 1. refresh V on every position and alternative rack (position-only pass, batched)
def pos_item(rec, rack):
    unseen = rec["unseen"] + rec["rack"] - rack      # tiles unseen to me if I held `rack` instead
    return helper.build_pos(rec["board"], rack, np.maximum(unseen, 0), rec["scores"], rec["bag"], rec["turn"])
@torch.no_grad()
def values(pairs, chunk=4096):
    """pairs: iterable of (rec, rack) -> (N, 2) [spread/100, P(win)], built and evaluated chunk by chunk."""
    out, buf = [], []
    def flush():
        bt = collate_moves(buf); bt = {k: v.to(dev) for k, v in bt.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): _, v = model.qv(bt, with_moves=False)
        v = v.float(); out.append(torch.stack([v[:, 0], torch.sigmoid(v[:, 1])], 1).cpu().numpy()); buf.clear()
    for rec, rack in pairs:
        buf.append(pos_item(rec, rack))
        if len(buf) == chunk: flush()
    if buf: flush()
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)
model.eval(); t0 = time.time()
flat = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti in range(len(tr))]
V = values((games[gi][0][ti], games[gi][0][ti]["rack"]) for gi, ti in flat)
alt_idx, n_alt = {}, 0
if LUCK:
    for gi, ti in flat:
        r = games[gi][0][ti]
        if "alt_racks" in r: alt_idx[(gi, ti)] = (n_alt, len(r["alt_racks"])); n_alt += len(r["alt_racks"])
VA = values((games[gi][0][ti], a) for gi, ti in flat if (gi, ti) in alt_idx for a in games[gi][0][ti]["alt_racks"]) if LUCK else None
vpos = {k: V[i] for i, k in enumerate(flat)}
print(f"refreshed {len(V)} positions + {n_alt} alternative racks in {time.time()-t0:.0f}s", flush=True)

# ---- 2. lambda-returns per player-sequence
targets = {}; luck_tot, spreads = [], []
for gi, (tr, sc) in enumerate(games):
    final = {0: sc[0] - sc[1], 1: sc[1] - sc[0]}; luck_g = {0: 0.0, 1: 0.0}
    for p in (0, 1):
        idx = [ti for ti, r in enumerate(tr) if r["mover"] == p]
        G = W = None
        for j in range(len(idx) - 1, -1, -1):
            ti = idx[j]; r = tr[ti]; sp = r["scores"][0] - r["scores"][1]
            if j == len(idx) - 1:
                G = (final[p] - sp) / 100; W = 1.0 if final[p] > 0 else 0.5 if final[p] == 0 else 0.0
            else:
                tn = idx[j + 1]; rn = tr[tn]; d = ((rn["scores"][0] - rn["scores"][1]) - sp) / 100
                vb = vpos[(gi, tn)]
                if (gi, tn) in alt_idx:
                    a, n = alt_idx[(gi, tn)]; mean = VA[a:a + n].mean(0); luck_g[p] += (vb[0] - mean[0]) * 100; vb = mean
                G = d + (1 - LAM) * vb[0] + LAM * G; W = (1 - LAM) * vb[1] + LAM * W
            targets[(gi, ti)] = (np.float32(G), np.float32(W))
    luck_tot.append(luck_g[0] - luck_g[1]); spreads.append(final[0])
luck_tot, spreads = np.array(luck_tot), np.array(spreads)
if LUCK and len(spreads) > 10:
    print(f"draw luck: per-game luck difference sd {luck_tot.std():.1f} pts, corr with final spread {np.corrcoef(luck_tot, spreads)[0,1]:.3f}; "
          f"spread sd {spreads.std():.1f} -> luck-adjusted sd {(spreads - luck_tot).std():.1f} (variance x{(spreads - luck_tot).var() / spreads.var():.2f})", flush=True)

# ---- 3. train Q (chosen move) and V (position)
items = [(gi, ti) for gi, ti in flat if games[gi][0][ti]["move"][0] in ("play", "exchange")]
class D(Dataset):
    def __init__(self, seed): self.rng = random.Random(seed)
    def __len__(self): return 10 ** 8
    def __getitem__(self, i):
        while True:
            gi, ti = items[self.rng.randrange(len(items))]; r = games[gi][0][ti]; m = r["move"]
            it = helper.build(r, 0.0, target_keep=np.array(m[1])) if m[0] == "exchange" else helper.build(r, 0.0, target_rcd=(m[1], m[2], m[3], m[5]))
            if it is None: continue
            G, W = targets[(gi, ti)]; it["value"] = G; it["value_k"] = W
            p = pos_item(r, r["rack"]); it.update({f"p_{k}": v for k, v in p.items()}); return it
def coll(batch):
    b = collate_moves(batch); bp = collate_moves([{k[2:]: v for k, v in it.items() if k.startswith("p_")} for it in batch]); return b, bp
dl = DataLoader(D(0), batch_size=B, num_workers=int(E("NWORK", 16)), collate_fn=coll, persistent_workers=True, prefetch_factor=4)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
model.train(); step = 0; la = None; t0 = time.time()
for b, bp in dl:
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}; bp = {k: v.to(dev, non_blocking=True) for k, v in bp.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        q, _ = model.qv(b); _, v = model.qv(bp, with_moves=False)
    q = q.float()[torch.arange(len(b["target"]), device=dev), b["target"]]; v = v.float()
    G, W = b["value"], b["value_k"]
    loss = F.smooth_l1_loss(q[:, 0], G) + F.binary_cross_entropy_with_logits(q[:, 1], W) + F.smooth_l1_loss(v[:, 0], G) + F.binary_cross_entropy_with_logits(v[:, 1], W)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 200 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}  |Q-G| {(q[:,0]-G).abs().mean().item()*100:.1f} pts  |V-G| {(v[:,0]-G).abs().mean().item()*100:.1f} pts", flush=True)
    if step >= STEPS: break
torch.save(model.state_dict(), f"data/policy_{TAG}.pt"); print(f"=== saved data/policy_{TAG}.pt", flush=True)
