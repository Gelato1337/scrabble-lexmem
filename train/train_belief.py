"""Step 4: learned belief over the opponent's rack. Supervised on replayed HastyBot games where both racks are known:
input = board, my rack, unseen counts, the opponent's last two plays (placed-tile planes) and how many tiles they kept;
target = the opponent's current 7 tiles (27 counts). Loss = multinomial cross-entropy. Baseline = uniform draw from unseen.
Env: TRAIN, TEST, STEPS, B, LR, TAG, NWORK"""
import os, sys, time, math, pickle, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; STEPS = int(E("STEPS", 6000)); B = int(E("B", 128)); LR = float(E("LR", 3e-4)); TAG = E("TAG", "belief")
NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
games_tr = pickle.load(open(f"data/games_{E('TRAIN','hasty_train')}.pkl", "rb")); games_te = pickle.load(open(f"data/games_{E('TEST','hasty_test')}.pkl", "rb"))

def hist_planes(traj, ti):
    """planes of the opponent's last two plays (letters placed, 1..26; blank as letter), and kept-tile counts."""
    planes = np.zeros((2, N, N), dtype=np.int64); kept = np.zeros(2, dtype=np.float32); k = 0
    for j in range(ti - 1, -1, -1):
        r = traj[j]
        if r["mover"] == traj[ti]["mover"]: continue
        if r["move"][0] == "play":
            for (rr, cc, l) in r["placed"]: planes[k, rr, cc] = l + 1
            kept[k] = 7 - len(r["placed"])
        elif r["move"][0] == "exchange": kept[k] = -1
        else: kept[k] = 7
        k += 1
        if k == 2: break
    return planes, kept

class BDS(Dataset):
    def __init__(self, games, n=10**7, seed=0, fixed=False):
        self.games, self.n, self.seed, self.rng, self.fixed = games, n, seed, None, fixed
        self.items = [(gi, ti) for gi, (traj, _) in enumerate(games) for ti in range(1, len(traj) - 1)]
        if fixed: self.items = self.items[:20000]
    def __len__(self): return len(self.items) if self.fixed else self.n
    def __getitem__(self, k):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + k)
        gi, ti = self.items[k] if self.fixed else self.items[self.rng.randrange(len(self.items))]
        traj = self.games[gi][0]; rec = traj[ti]; nxt = traj[ti + 1]
        opp = nxt["rack"].astype(np.float32)            # the opponent's rack right now (they move next)
        planes, kept = hist_planes(traj, ti)
        return dict(board=rec["board"].astype(np.int64) + 1, rack=rec["rack"].astype(np.float32), unseen=rec["unseen"].astype(np.float32),
                    glob=np.array([rec["bag"] / 100, (rec["scores"][0] - rec["scores"][1]) / 100, rec["turn"] / 30], dtype=np.float32),
                    hist=planes, kept=kept, target=opp)

def collate(items): return {k: torch.tensor(np.stack([it[k] for it in items])) for k in items[0]}

class BeliefNet(nn.Module):
    def __init__(s, D=256):
        super().__init__()
        s.bemb = nn.Embedding(27, 24); s.hemb = nn.Embedding(27, 8)
        s.cnn = nn.Sequential(nn.Conv2d(24 + 16 + 5, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                              nn.Conv2d(64, 16, 1), nn.GELU(), nn.Flatten(), nn.Linear(16 * 225, D))
        s.register_buffer("prem", F.one_hot(torch.tensor(PREM), 5).permute(2, 0, 1).float()[None])
        s.mlp = nn.Sequential(nn.Linear(D + 27 + 27 + 3 + 2, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 27))
    def forward(s, b):
        B_ = b["rack"].shape[0]
        x = torch.cat([s.bemb(b["board"]).permute(0, 3, 1, 2), s.hemb(b["hist"][:, 0]).permute(0, 3, 1, 2), s.hemb(b["hist"][:, 1]).permute(0, 3, 1, 2), s.prem.expand(B_, -1, -1, -1)], 1)
        h = torch.cat([s.cnn(x), b["rack"] / 3, b["unseen"] / 12, b["glob"], b["kept"] / 7], 1)
        logits = s.mlp(h)
        return logits.masked_fill(b["unseen"] <= 0, -1e4)       # a tile not in the unseen pool cannot be on their rack

model = BeliefNet().to(dev); print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
dl = DataLoader(BDS(games_tr, seed=0), batch_size=B, num_workers=NW_, collate_fn=collate, persistent_workers=NW_ > 0)
tdl = DataLoader(BDS(games_te, fixed=True), batch_size=256, num_workers=min(NW_, 8), collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
def nll(logits, target):     # multinomial cross-entropy per tile
    return -(target * F.log_softmax(logits, 1)).sum(1) / target.sum(1).clamp_min(1)

@torch.no_grad()
def evaluate(tag):
    model.eval(); n_net, n_base, tv_net, tv_base = [], [], [], []
    for b in tdl:
        b = {k: v.to(dev) for k, v in b.items()}
        logits = model(b); base = torch.log(b["unseen"].clamp_min(1e-9)) - torch.log(b["unseen"].sum(1, keepdim=True))
        n_net += nll(logits.float(), b["target"]).tolist(); n_base += nll(base, b["target"]).tolist()
        p = torch.softmax(logits.float(), 1) * 7; q = b["unseen"] / b["unseen"].sum(1, keepdim=True) * 7
        tv_net += (0.5 * (p - b["target"]).abs().sum(1)).tolist(); tv_base += (0.5 * (q - b["target"]).abs().sum(1)).tolist()
    model.train()
    print(f"  [{tag}] per-tile NLL net={np.mean(n_net):.4f} uniform-from-unseen={np.mean(n_base):.4f} | expected-rack L1 error (tiles) net={np.mean(tv_net):.3f} baseline={np.mean(tv_base):.3f}", flush=True)
    return float(np.mean(n_net)), float(np.mean(n_base))

evaluate("untrained"); t0 = time.time(); step = 0; la = None
for b in dl:
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    loss = nll(model(b).float(), b["target"]).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}", flush=True)
    if step % int(E("EVAL_EVERY", 1000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/belief_{TAG}.pt")
    if step >= STEPS: break
r = evaluate("final"); pickle.dump(r, open(f"data/result_belief_{TAG}.pkl", "wb"))
