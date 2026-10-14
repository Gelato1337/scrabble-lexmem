"""Branch-only test: can the shared-draw branch pairs (selfplay_v2.py) teach the net anything beyond what it already prefers?
Train the policy only on the branch pairs of the training games (gi % HOLD != 0):
  BCE(sigmoid(l_a - l_b), sigmoid(dU / TAU_B)) + BETA_KL * KL(pi_start || pi)
for EPOCHS passes; after each pass, held-out pairs (gi % HOLD == 0): sign agreement of l_a - l_b with dU (0.5 = no signal) and the
correlation, overall and by phase. Learning beyond the start net = held-out agreement rising above the start net's.
Usage: TRAIN=v2_w_g2_0,... INIT=data/policy_v2_w_g1.pt TAG=v2_w_branch EPOCHS=5 PYTHONPATH=. python train/train_branch.py"""
import os, time, math, copy, pickle, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
HOLD = int(E("HOLDOUT_MOD", 10)); EPOCHS = int(E("EPOCHS", 5)); B = int(E("B", 48)); LR = float(E("LR", 5e-5)); BETA_KL = float(E("BETA_KL", 1.0))
TAU_B = float(E("TAU_B", 0.25)); TAG = E("TAG", "v2_branch"); NW = int(E("NWORK", 16))
torch.manual_seed(0); random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex).to(dev)
model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); anchor = copy.deepcopy(model).eval()
for p in anchor.parameters(): p.requires_grad_(False)
games = [g for t in E("TRAIN").split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]
pairs = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti, r in enumerate(tr) if "branch" in r and r["move"][0] in ("play", "exchange")]
train = [k for k in pairs if k[0] % HOLD != 0]; held = [k for k in pairs if k[0] % HOLD == 0]
print(f"{len(games)} games; branch pairs: {len(train)} train, {len(held)} held out", flush=True)
helper = MoveDataset(lex, [], exch=True)
def same(m, b): return (m[3] == 2 and b[3] == 2 and tuple(m[5]) == tuple(b[5])) or (m[3] != 2 and b[3] != 2 and tuple(m[1:6]) == tuple(b[1:6]))
def phase(r):
    u = int(r["unseen"].sum()); return "pre-endgame/endgame" if u <= 14 else "late (8-30)" if r["bag"] <= 30 else "early/middle (31+)"
def item(k):
    gi, ti = k; r = games[gi][0][ti]; m = r["move"]
    it = helper.build(r, 0.0, target_keep=np.array(m[1])) if m[0] == "exchange" else helper.build(r, 0.0, target_rcd=(m[1], m[2], m[3], m[5]))
    if it is None: return None
    bm, dU = r["branch"]; j = next((j for j, x in enumerate(it["_moves"]) if same(x, bm)), -1)
    if j < 0: return None
    it["bidx"], it["dU"], it["ph"] = j, np.float32(dU), phase(r); return it
class DS(Dataset):
    def __init__(self, keys, shuffle_seed=None):
        self.keys = list(keys)
        if shuffle_seed is not None: random.Random(shuffle_seed).shuffle(self.keys)
    def __len__(self): return len(self.keys)
    def __getitem__(self, i): return item(self.keys[i])
def coll(batch):
    batch = [x for x in batch if x is not None]; b = collate_moves(batch)
    b["bidx"] = torch.tensor([x["bidx"] for x in batch]); b["dU"] = torch.tensor([x["dU"] for x in batch]); return b, [x["ph"] for x in batch]
def to(b): return {k: v.to(dev, non_blocking=True) for k, v in b.items()}
@torch.no_grad()
def check(net, tag):
    net.eval(); d, u, ph = [], [], []
    for b, p in DataLoader(DS(held), batch_size=64, num_workers=NW, collate_fn=coll):
        b = to(b)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, _ = net(b)
        pl = pl.float(); ar = torch.arange(len(b["target"]), device=dev)
        d += (pl[ar, b["target"]] - pl[ar, b["bidx"]]).tolist(); u += b["dU"].tolist(); ph += p
    d, u, ph = np.array(d), np.array(u), np.array(ph); nz = u != 0
    line = f"[{tag}] held-out n={len(d)}: sign agreement {np.mean(np.sign(d[nz]) == np.sign(u[nz])):.3f}, corr {np.corrcoef(d, u)[0,1]:+.3f}"
    for p_ in ("early/middle (31+)", "late (8-30)", "pre-endgame/endgame"):
        k = (ph == p_) & nz
        if k.sum() > 30: line += f" | {p_}: {np.mean(np.sign(d[k]) == np.sign(u[k])):.3f}"
    print(line, flush=True); net.train()
check(anchor, "start net")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); t0 = time.time(); step = 0
for ep in range(EPOCHS):
    la = None
    for b, _ in DataLoader(DS(train, ep), batch_size=B, num_workers=NW, collate_fn=coll, prefetch_factor=4):
        b = to(b)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pl, _ = model(b)
            with torch.no_grad(): pa, _ = anchor(b)
        pl, pa = pl.float(), pa.float(); ar = torch.arange(len(b["target"]), device=dev)
        l_br = F.binary_cross_entropy_with_logits(pl[ar, b["target"]] - pl[ar, b["bidx"]], torch.sigmoid(b["dU"] / TAU_B))
        lpa = torch.log_softmax(pa, 1); l_kl = (lpa.exp() * (lpa - torch.log_softmax(pl, 1))).sum(1).mean()
        loss = l_br + BETA_KL * l_kl; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; la = (l_br.item(), l_kl.item()) if la is None else (0.98 * la[0] + 0.02 * l_br.item(), 0.98 * la[1] + 0.02 * l_kl.item())
        if step % 500 == 0: print(f"epoch {ep+1} step {step} t={time.time()-t0:.0f}s branch {la[0]:.4f} kl {la[1]:.4f}", flush=True)
    check(model, f"after epoch {ep+1}"); torch.save(model.state_dict(), f"data/policy_{TAG}.pt")
print(f"=== saved data/policy_{TAG}.pt", flush=True)
