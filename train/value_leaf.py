"""Our own search leaf (round 5): a copy of a policy net whose TD value head learns the search's win probability.
go/lexmem_search LEAF=win then ends every playout with this net's estimate (served by selfplay/server_infer.py as VCKPT, the
policy itself unchanged) instead of Macondo's win-% table + leave values.
Data: DATA=a,b -> data/vleaf_a.pkl, ... (selfplay/ei_data.py values): position (mover's view) -> y = the search's win
probability for the mover. Model: INIT = the start net, every weight fine-tuned; target head = vh[:, 1] (win logit; vh[:, 0]
is not trained), loss = binary cross-entropy against the soft target y. Held out: positions of games with g % 20 == 0 in
each file. Env: DATA INIT USE_LEX RETR STEPS (6000) B (64) LR (1e-4) EVAL_EVERY (1000) TAG -> data/value_<TAG>.pt (the step
with the lowest held-out loss). Prints held-out MAE in win-% points against the constant predictor (training mean)."""
import os, math, time, pickle, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import Lexicon
from lexmem.policy import PolicyNet, MoveDataset, collate_moves
torch.backends.mha.set_fastpath_enabled(False)
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_LEX = E("USE_LEX", "0") == "1"; STEPS = int(E("STEPS", 6000)); B = int(E("B", 64)); LR = float(E("LR", 1e-4)); TAG = E("TAG", "vleaf")
NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8)))
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))
rows = []
for name in E("DATA").split(","): rows += pickle.load(open(f"data/vleaf_{name}.pkl", "rb"))
tr = [r for r in rows if r["g"] % 20]; te = [r for r in rows if r["g"] % 20 == 0]
print(f"{len(tr)} training / {len(te)} held-out positions from {E('DATA')}", flush=True)

class VDS(Dataset):
    def __init__(self, rows): self.rows, self.h = rows, MoveDataset(lex, [], use_lex=USE_LEX)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; c = r["rec"]
        it = self.h.build_pos(c["board"], c["rack"], c["unseen"], c["scores"], c["bag"], c["turn"]); it["y"] = np.float32(r["y"]); return it
def collate(items):
    out = collate_moves(items); out["y"] = torch.tensor(np.array([it["y"] for it in items])); return out

model = PolicyNet(lex, use_lex=USE_LEX).to(dev)
if USE_LEX:
    if E("RETR"): model.load_retrieval(E("RETR"))
    else: model.refresh_keys()
model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); print("init from", E("INIT"), flush=True)
dl = DataLoader(VDS(tr), batch_size=B, shuffle=True, drop_last=True, num_workers=NW_, collate_fn=collate, persistent_workers=NW_ > 0)
EV = list(DataLoader(VDS(te), batch_size=256, num_workers=NW_, collate_fn=collate))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
ymean = float(np.mean([r["y"] for r in tr]))

def win_logit(b):
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): _, v = model.qv(b, with_moves=False)
    return v[:, 1].float()

@torch.no_grad()
def evaluate(tag):
    model.eval(); bce = mae = mae0 = n = 0.0
    for b in EV:
        b = {k: v.to(dev) for k, v in b.items()}; z = win_logit(b); y = b["y"]
        bce += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item(); mae += (torch.sigmoid(z) - y).abs().sum().item()
        mae0 += (ymean - y).abs().sum().item(); n += len(y)
    model.train(); print(f"  [{tag}] held-out BCE {bce / n:.4f}, MAE {100 * mae / n:.2f} win-% points (constant {100 * mae0 / n:.2f})", flush=True)
    return bce / n

evaluate("untrained"); best = float("inf"); step = 0; t0 = time.time(); la = None
while step < STEPS:
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        loss = F.binary_cross_entropy_with_logits(win_logit(b), b["y"])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
        if step % 100 == 0: print(f"step {step} t={time.time() - t0:.0f}s loss={la:.4f}", flush=True)
        if step % int(E("EVAL_EVERY", 1000)) == 0 or step == STEPS:
            l = evaluate(f"step {step}")
            if l < best: best = l; torch.save(model.state_dict(), f"data/value_{TAG}.pt"); print(f"  saved data/value_{TAG}.pt", flush=True)
        if step >= STEPS: break
print(f"best held-out BCE {best:.4f}")
