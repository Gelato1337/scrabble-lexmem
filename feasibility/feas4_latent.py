"""
Step 4 feasibility: MuZero-style latent dynamics WITH the lexicon layer as the prediction head.

  h(o_t)              -> s_t              representation (board, rack, unseen, scores)
  g(s_t, a_t, c_t)    -> s_{t+1}          dynamics; a_t = tiles placed (15x15x27 plane) or exchange; c_t = tiles drawn (27 counts)
                                          one step = one half-move (mover alternates); s carries "whose turn"
  f(s, lane)          -> logits over words  the lexicon layer, query built from the LATENT + lane tokens
  v(s)                -> final spread (aux)

Training: unroll K steps from real trajectories with the TRUE actions and draws; at every unrolled step,
  retrieval loss on a random lane of the true position o_{t+k} using the predicted latent;
  + consistency loss  cos(g(...), stopgrad h(o_{t+k}))  (EfficientZero-style).
The gate: retrieval quality from the k-step-imagined latent vs from h(o_{t+k}) directly, as a function of k.
If the latent smooths away lexicon-level detail, the k>0 curves collapse toward the untrained baseline.

Env: STEPS, B, K, TAG, INIT (path to a step-2 checkpoint to init the word encoder)
"""
import os, sys, time, pickle, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.data import *; from lexmem.model import WordEncoder
import warnings; warnings.filterwarnings("ignore")
STEPS = int(os.environ.get("STEPS", 3000)); B = int(os.environ.get("B", 32)); K = int(os.environ.get("K", 4))
TAG = os.environ.get("TAG", "latent"); INIT = os.environ.get("INIT", ""); D = 256; NW_ = int(os.environ.get("NWORK", os.environ.get("SLURM_CPUS_PER_TASK", 12)))
LR = float(os.environ.get("LR", 3e-4)); ALPHA = 0.5
torch.manual_seed(0); np.random.seed(0); random.seed(0); dev = "cuda"
lex = Lexicon(os.environ.get("LEXICON", os.path.join(ROOT, "enable1.txt"))); heldout = holdout_mask(lex)
games_tr = pickle.load(open(f"data/games_{os.environ.get('TRAIN','train')}.pkl", "rb")); games_te = pickle.load(open(f"data/games_{os.environ.get('TEST','test')}.pkl", "rb"))
NEV = int(os.environ.get("NEV", 200))

# ------------------------------------------------------------------ data: K-step windows
def obs_of(rec):
    return dict(board=rec["board"].astype(np.int64) + 1, rack=rec["rack"].astype(np.float32), unseen=rec["unseen"].astype(np.float32),
                glob=np.array([rec["bag"] / 100, (rec["scores"][0] - rec["scores"][1]) / 100, rec["turn"] / 30, rec["mover"]], dtype=np.float32))

def action_of(rec):
    plane = np.zeros((N, N), dtype=np.int64)          # 0 = nothing, 1..26 letter placed (blank -> letter too)
    if rec["move"][0] == "play":
        for (r, c, l) in rec["placed"]: plane[r, c] = l + 1
    kind = {"play": 0, "exchange": 1, "pass": 2}[rec["move"][0]]
    drawn = np.zeros(27, dtype=np.float32)
    for t in rec["drawn"]: drawn[t] += 1
    return dict(aplane=plane, akind=kind, drawn=drawn)

class WindowDataset(Dataset):
    def __init__(self, games, n_items=10**7, seed=0, n_neg=384, n_pos=192):
        self.games = games; self.n = n_items; self.seed = seed; self.rng = None; self.n_neg = n_neg; self.n_pos = n_pos
        self.starts = [(gi, ti) for gi, (traj, _) in enumerate(games) for ti in range(len(traj) - K)]
        self.cache = {}
    def __len__(self): return self.n
    def _lanes(self, gi, ti):
        key = (gi, ti)
        if key not in self.cache:
            if len(self.cache) > 128: self.cache.clear()
            rec = self.games[gi][0][ti]
            self.cache[key] = (cross_checks(lex, rec["board"], rec["blanks"]), lanes(rec["board"]))
        return self.cache[key]
    def lane_item(self, gi, ti):
        """random lane of position (gi,ti) with its reachable label from the mover's perspective."""
        rng = self.rng; rec = self.games[gi][0][ti]; cc, ll = self._lanes(gi, ti)
        for _ in range(30):
            r, c, d, L = ll[rng.randrange(len(ll))]
            info = lane_info(rec["board"], rec["blanks"], cc, r, c, d, L)
            g, loc = lane_candidates(lex, info, L)
            if len(loc) < 2: continue
            ok, miss, P, score, w = reachable_label(lex, info, L, loc, rec["rack"], rec["unseen"])
            keepm = ~heldout[g]; pos = np.where(ok & keepm)[0]; neg = np.where(~ok & keepm)[0]
            if len(pos) == 0: continue
            if len(pos) > self.n_pos: pos = np.array(rng.sample(pos.tolist(), self.n_pos))
            if len(neg) > self.n_neg: neg = np.array(rng.sample(neg.tolist(), self.n_neg))
            sel = np.concatenate([pos, neg])
            return dict(lane=lane_tokens(info, L), L=L, cand=g[sel].astype(np.int64), weight=w[sel].astype(np.float32),
                        ok=ok[sel], miss=miss[sel].astype(np.int64), masksize=len(loc))
        return None
    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + i)
        rng = self.rng
        while True:
            gi, ti = self.starts[rng.randrange(len(self.starts))]
            traj, scores = self.games[gi]
            steps = []
            good = True
            for k in range(K + 1):
                rec = traj[ti + k]
                li = self.lane_item(gi, ti + k)
                if li is None: good = False; break
                final = (scores[rec["mover"]] - scores[1 - rec["mover"]]) / 100
                steps.append(dict(obs=obs_of(rec), act=action_of(rec), lane=li, value=np.float32(final)))
            if good: return steps

def collate_w(items):
    out = []
    for k in range(K + 1):
        its = [it[k] for it in items]; B_ = len(its); C = max(len(x["lane"]["cand"]) for x in its)
        d = {kk: torch.tensor(np.stack([x["obs"][kk] for x in its])) for kk in ("board", "rack", "unseen", "glob")}
        d["aplane"] = torch.tensor(np.stack([x["act"]["aplane"] for x in its])); d["akind"] = torch.tensor([x["act"]["akind"] for x in its])
        d["drawn"] = torch.tensor(np.stack([x["act"]["drawn"] for x in its])); d["value"] = torch.tensor([x["value"] for x in its])
        d["lane"] = torch.tensor(np.stack([x["lane"]["lane"] for x in its])); d["L"] = torch.tensor([x["lane"]["L"] for x in its])
        cand = torch.zeros(B_, C, dtype=torch.long); valid = torch.zeros(B_, C, dtype=torch.bool); weight = torch.zeros(B_, C)
        ok = torch.zeros(B_, C, dtype=torch.bool); miss = torch.full((B_, C), -1, dtype=torch.long)
        for b, x in enumerate(its):
            n = len(x["lane"]["cand"]); cand[b, :n] = torch.tensor(x["lane"]["cand"]); valid[b, :n] = True
            weight[b, :n] = torch.tensor(x["lane"]["weight"]); ok[b, :n] = torch.tensor(x["lane"]["ok"]); miss[b, :n] = torch.tensor(x["lane"]["miss"])
        d.update(cand=cand, valid=valid, weight=weight, ok=ok, miss=miss, masksize=torch.tensor([x["lane"]["masksize"] for x in its]))
        out.append(d)
    return out

# ------------------------------------------------------------------ model
class Repr(nn.Module):
    def __init__(s):
        super().__init__()
        s.bemb = nn.Embedding(27, 32)
        s.cnn = nn.Sequential(nn.Conv2d(32 + 5, 96, 3, padding=1), nn.GELU(), nn.Conv2d(96, 96, 3, padding=1), nn.GELU(),
                              nn.Conv2d(96, 96, 3, padding=1), nn.GELU(), nn.Conv2d(96, 16, 1), nn.GELU(), nn.Flatten(), nn.Linear(16 * 225, D))
        s.rack = nn.Linear(27, D); s.unseen = nn.Linear(27, D); s.glob = nn.Linear(4, D)
        s.mlp = nn.Sequential(nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D), nn.LayerNorm(D))
        s.register_buffer("prem", F.one_hot(torch.tensor(PREM), 5).permute(2, 0, 1).float()[None])
    def forward(s, o):
        b = s.bemb(o["board"]).permute(0, 3, 1, 2)
        x = torch.cat([b, s.prem.expand(b.shape[0], -1, -1, -1)], 1)
        return s.mlp(torch.cat([s.cnn(x), s.rack(o["rack"] / 3), s.unseen(o["unseen"] / 12), s.glob(o["glob"])], 1))

class Dyn(nn.Module):
    def __init__(s):
        super().__init__()
        s.aemb = nn.Embedding(27, 16); s.acnn = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 8, 1), nn.GELU(), nn.Flatten(), nn.Linear(8 * 225, D))
        s.kind = nn.Embedding(3, D); s.drawn = nn.Linear(27, D)
        s.mlp = nn.Sequential(nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        s.norm = nn.LayerNorm(D)
    def forward(s, state, act):
        a = s.acnn(s.aemb(act["aplane"]).permute(0, 3, 1, 2)) + s.kind(act["akind"])
        return s.norm(state + s.mlp(torch.cat([state, a, s.drawn(act["drawn"] / 3), state * a], 1)))

class LaneHead(nn.Module):
    """query = transformer over [state token, lane tokens] -> D"""
    def __init__(s):
        super().__init__()
        s.lane = nn.Linear(LANE_F, D); s.lanepos = nn.Parameter(torch.randn(MAXL, D) * 0.02); s.st = nn.Linear(D, D)
        enc = nn.TransformerEncoderLayer(D, 4, D * 4, dropout=0.0, batch_first=True, norm_first=True)
        s.tr = nn.TransformerEncoder(enc, 2); s.norm = nn.LayerNorm(D); s.scale = nn.Parameter(torch.tensor(D ** -0.5))
    def forward(s, state, lane, L):
        x = torch.cat([s.st(state)[:, None], s.lane(lane) + s.lanepos], 1)
        pad = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        pad[:, 1:] = torch.arange(MAXL, device=x.device)[None] >= L[:, None]
        return s.norm(s.tr(x, src_key_padding_mask=pad)[:, 0])

class MuLex(nn.Module):
    def __init__(s):
        super().__init__()
        s.h = Repr(); s.g = Dyn(); s.head = LaneHead(); s.wenc = WordEncoder(lex, D)
        s.v = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        s.proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))    # consistency projector
    def retrieve(s, state, d, keys=None):
        q = s.head(state, d["lane"], d["L"])
        k = s.wenc(d["cand"]) if keys is None else keys[d["cand"]]
        return (torch.einsum("bd,bcd->bc", q, k) * s.head.scale).masked_fill(~d["valid"], -1e4)

def list_loss(logits, d):
    valid = d["valid"]; w = d["weight"] * valid; ok = d["ok"].float() * valid
    target = ALPHA * w / w.sum(1, keepdim=True).clamp_min(1e-9) + (1 - ALPHA) * ok / ok.sum(1, keepdim=True).clamp_min(1e-9)
    return -(target * F.log_softmax(logits.float(), 1)).sum(1).mean()

model = MuLex().to(dev)
if INIT:
    sd = torch.load(INIT, map_location=dev); model.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.")})
    print("word encoder initialised from", INIT)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  K={K}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
dl = DataLoader(WindowDataset(games_tr, seed=0), batch_size=B, num_workers=NW_, collate_fn=collate_w, persistent_workers=True, prefetch_factor=4)

ev_ds = WindowDataset(games_te, seed=7, n_neg=10**6, n_pos=10**6); ev_ds.rng = random.Random(7)
EV = [ev_ds[i] for i in range(NEV)]
print("eval windows built", flush=True)

@torch.no_grad()
def full_keys():
    ks = []
    for i in range(0, lex.NW, 8192):
        with torch.autocast("cuda", dtype=torch.bfloat16): ks.append(model.wenc(torch.arange(i, min(i + 8192, lex.NW), device=dev)).float())
    return torch.cat(ks)

def ndcg(g, ideal, k=32):
    disc = 1 / np.log2(np.arange(2, k + 2)); dcg = (g[:k] * disc[:len(g[:k])]).sum(); idcg = (ideal[:k] * disc[:len(ideal[:k])]).sum()
    return dcg / idcg if idcg > 0 else np.nan

@torch.no_grad()
def evaluate(tag):
    """For each window: latent from h(o_k) (direct) vs from k-step imagination g(...g(h(o_0),a_0)...). Metrics per k."""
    model.eval(); keys = full_keys()
    res = {k: dict(nd_direct=[], nd_imag=[], pct_direct={m: [] for m in range(4)}, pct_imag={m: [] for m in range(4)}, cos=[]) for k in range(K + 1)}
    verr = []
    for w in EV:
        ds_ = collate_w([w]); ds_ = [{kk: v.to(dev) for kk, v in d.items()} for d in ds_]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model.h(ds_[0])
            for k in range(K + 1):
                d = ds_[k]; sd = model.h(d)
                lg_i = model.retrieve(s, d, keys)[0].float().cpu().numpy(); lg_d = model.retrieve(sd, d, keys)[0].float().cpu().numpy()
                if k == 0: verr.append(abs(model.v(s).item() - d["value"].item()))
                n = int(d["valid"].sum()); wt = d["weight"][0, :n].cpu().numpy(); ok = d["ok"][0, :n].cpu().numpy(); miss = d["miss"][0, :n].cpu().numpy()
                ideal = np.sort(wt)[::-1]
                for name, lg in (("direct", lg_d[:n]), ("imag", lg_i[:n])):
                    order = np.argsort(-lg, kind="stable"); res[k][f"nd_{name}"].append(ndcg(wt[order], ideal))
                    rank = np.empty(n); rank[order] = np.arange(n); pctl = rank / max(n - 1, 1)
                    for m in range(4):
                        sel = ok & (miss == m)
                        if sel.any(): res[k][f"pct_{name}"][m].append(pctl[sel].mean())
                res[k]["cos"].append(F.cosine_similarity(s.float(), sd.float()).item())
                s = model.g(s, d)
    model.train()
    print(f"  [{tag}] value MAE={np.mean(verr):.3f} (spread/100)")
    for k in range(K + 1):
        r = res[k]
        pd = " ".join(f"m{m}={np.mean(r['pct_direct'][m]):.2f}" for m in range(4) if r['pct_direct'][m])
        pi = " ".join(f"m{m}={np.mean(r['pct_imag'][m]):.2f}" for m in range(4) if r['pct_imag'][m])
        print(f"    k={k}: NDCG direct={np.nanmean(r['nd_direct']):.3f} imagined={np.nanmean(r['nd_imag']):.3f} | pct direct [{pd}] imagined [{pi}] | cos(imag,direct)={np.mean(r['cos']):.3f}", flush=True)
    return res

evaluate("untrained")
t0 = time.time(); step = 0; la = None
for win in dl:
    win = [{kk: v.to(dev, non_blocking=True) for kk, v in d.items()} for d in win]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = model.h(win[0]); loss = 0.0; parts = []
        for k in range(K + 1):
            d = win[k]
            lr_ = list_loss(model.retrieve(s, d), d)
            lv = F.smooth_l1_loss(model.v(s).squeeze(1).float(), d["value"])
            lc = 0.0
            if k > 0:
                with torch.no_grad(): tgt = model.h(d).float()
                lc = (1 - F.cosine_similarity(model.proj(s).float(), tgt)).mean()
            loss = loss + lr_ + 0.5 * lv + 1.0 * lc; parts.append((lr_.item(), float(lc)))
            if k < K: s = model.g(s, d)
        loss = loss / (K + 1)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
    if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.3f} retr[k]={[round(p[0],2) for p in parts]} cons[k]={[round(p[1],2) for p in parts]}", flush=True)
    if step % int(os.environ.get("EVAL_EVERY", 1000)) == 0 or step == STEPS:
        evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/mulex_{TAG}.pt")
    if step >= STEPS: break
print("FINAL"); R = evaluate("final"); pickle.dump(R, open(f"data/result_{TAG}.pkl", "wb"))
