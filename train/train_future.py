"""Future-word α-target for the lexicon layer, with an optional recurrent lattice projection (Lattice Deduction Transformer style).
Unit = (post-move state of a candidate, side, lane). Candidate set = exact rules mask for that lane on the post-move board
(pattern + cross-checks + letters available: me = leave+unseen with <=3 missing, opponent = within unseen).
α target = words actually played in that lane by that side across the sim's sampled worlds, weighted by frequency.
Model: R rounds. Round r scores the alive words; keep the top 25 % (>=16); the survivors' pooled keys refine the query. Every
round is supervised (deep supervision); eliminated words can never return (monotone lattice descent).
Env: LABELS (comma list), R, STEPS, B, LR, TAG, RETR (init), NWORK"""
import os, sys, time, math, pickle, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.data import lane_tokens, LANE_F, MAXL; from lexmem.model import WordEncoder, QueryEncoder
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; R = int(E("R", 3)); STEPS = int(E("STEPS", 12000)); B = int(E("B", 48)); LR = float(E("LR", 2e-4)); TAG = E("TAG", "fut")
NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); NNEG = int(E("NNEG", 512)); KEEP = float(E("KEEP", 0.25)); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))
DIR = ((0, 1), (1, 0))

def build_unit(x, ci, side, lane_rows, rng=None, nneg=None):
    """lane_rows: fut rows (ci,side,r,c,d,L,widx,count) for one lane. Returns an item dict or None."""
    s, r0, c0, d0, L0, w0 = x["cands"][ci][:6]
    b2, bl2, leave, _ = apply_move(lex, x["board"], x["blanks"], x["rack"], r0, c0, d0, L0, w0)
    unseen = x["unseen"]
    r, c, d, L = (int(v) for v in lane_rows[0][2:6])
    if (r, c, d, L) not in set(lanes(b2)): return None
    cc = cross_checks(lex, b2, bl2); info = lane_info(b2, bl2, cc, r, c, d, L)
    g, loc = lane_candidates(lex, info, L)
    if len(loc) < 2: return None
    need = need_counts(lex, info, L, loc)
    if side == 0:
        deficit = np.maximum(need - leave[:26][None], 0)
        ok = (np.maximum(deficit - unseen[:26][None], 0).sum(1) <= int(leave[26]) + int(unseen[26])) & (deficit.sum(1) - int(leave[26]) <= 3)
    else:
        ok = np.maximum(need - unseen[:26][None], 0).sum(1) <= int(unseen[26])
    g = g[ok]; loc = loc[ok]
    if len(g) < 2: return None
    tgt = np.zeros(len(g), dtype=np.float32); pos_of = {int(w): i for i, w in enumerate(g)}
    for row in lane_rows:
        i = pos_of.get(int(row[6]))
        if i is not None: tgt[i] += row[7]
    if tgt.sum() == 0: return None
    score = lane_scores(lex, info, L, loc).astype(np.float32)
    if nneg is not None and len(g) > nneg + (tgt > 0).sum():
        pos = np.where(tgt > 0)[0]; neg = np.where(tgt == 0)[0]; neg = np.array(rng.sample(neg.tolist(), nneg))
        sel = np.concatenate([pos, neg]); g, tgt, score = g[sel], tgt[sel], score[sel]
    return dict(rack=leave.astype(np.float32), unseen=unseen.astype(np.float32),
                glob=np.array([x["bag"] / 100, (x["scores"][0] + s - x["scores"][1]) / 100, L / 15, len(loc) / 1000, float(side)], dtype=np.float32),
                lane=lane_tokens(info, L), board=b2.astype(np.int64) + 1, L=L, cand=g.astype(np.int64), tgt=tgt / tgt.sum(), score=score,
                side=side, masksize=len(loc))

def units_of(x):
    """list of (ci, side, rows) grouped by lane."""
    f = x.get("fut")
    if f is None or len(f) == 0: return []
    out = {}
    for row in f: out.setdefault((int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])), []).append(row)
    return [(k[0], k[1], v) for k, v in out.items()]

class FDS(Dataset):
    def __init__(self, labels, n=10**7, seed=0):
        self.labels, self.n, self.seed, self.rng = labels, n, seed, None
    def __len__(self): return self.n
    def __getitem__(self, k):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + k)
        while True:
            x = self.labels[self.rng.randrange(len(self.labels))]; us = units_of(x)
            if not us: continue
            ci, side, rows = us[self.rng.randrange(len(us))]
            it = build_unit(x, ci, side, rows, self.rng, NNEG)
            if it is not None: return it

def collate(items):
    B_ = len(items); C = max(len(it["cand"]) for it in items)
    out = {k: torch.tensor(np.stack([it[k] for it in items])) for k in ("rack", "unseen", "glob", "lane", "board")}
    out["L"] = torch.tensor([it["L"] for it in items]); out["side"] = torch.tensor([it["side"] for it in items])
    cand = torch.zeros(B_, C, dtype=torch.long); valid = torch.zeros(B_, C, dtype=torch.bool); tgt = torch.zeros(B_, C); score = torch.zeros(B_, C)
    for b, it in enumerate(items):
        n = len(it["cand"]); cand[b, :n] = torch.tensor(it["cand"]); valid[b, :n] = True; tgt[b, :n] = torch.tensor(it["tgt"]); score[b, :n] = torch.tensor(it["score"])
    out.update(cand=cand, valid=valid, tgt=tgt, score=score); return out

class LatticeLex(nn.Module):
    def __init__(self, D=256, R=3):
        super().__init__()
        self.D, self.R = D, R; self.wenc = WordEncoder(lex, D, slot=False); self.qenc = QueryEncoder(D, 3)
        self.scale = nn.Parameter(torch.tensor(D ** -0.5))
        self.refine = nn.Sequential(nn.Linear(3 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D)); self.rnorm = nn.LayerNorm(D)
    def forward(self, b, keys=None):
        q = self.qenc(b["rack"], b["unseen"], b["glob"], b["lane"], b["board"], b["L"])
        k = self.wenc(b["cand"]) if keys is None else keys[b["cand"]]
        alive = b["valid"].clone(); outs = []
        for r in range(self.R):
            logits = torch.einsum("bd,bcd->bc", q, k) * self.scale
            outs.append((logits.masked_fill(~alive, -1e4), alive.clone()))
            if r < self.R - 1:
                with torch.no_grad():   # lattice projection: hard, monotone elimination
                    n_alive = alive.sum(1); n_keep = torch.clamp(torch.ceil(n_alive.float() * KEEP).long(), min=16); n_keep = torch.minimum(n_keep, n_alive)
                    srt = logits.masked_fill(~alive, -1e9).sort(1, descending=True).values
                    thr = srt.gather(1, (n_keep - 1).clamp_min(0)[:, None]); alive = alive & (logits >= thr)
                w = torch.softmax(logits.masked_fill(~alive, -1e4), 1); pooled = torch.einsum("bc,bcd->bd", w, k)
                q = self.rnorm(q + self.refine(torch.cat([q, pooled, q * pooled], 1)))
        return outs
    def rank_key(self, outs):
        key = torch.full_like(outs[0][0], -1e9)
        for r, (lg, al) in enumerate(outs): key = torch.where(al, r * 1e5 + lg, key)
        return key

def ndcg(g, ideal, k=32):
    disc = 1 / np.log2(np.arange(2, k + 2)); a = (g[:k] * disc[:len(g[:k])]).sum(); b = (ideal[:k] * disc[:len(ideal[:k])]).sum()
    return a / b if b > 0 else np.nan

def metrics(key, it):
    t = it["tgt"]; order = np.argsort(-key, kind="stable"); n = len(t)
    rank = np.empty(n); rank[order] = np.arange(n); pct = rank / max(n - 1, 1)
    return dict(m8=t[order[:8]].sum(), m32=t[order[:32]].sum(), ndcg=ndcg(t[order], np.sort(t)[::-1]), pct=(t * pct).sum())

if __name__ == "__main__":
    labels = []
    for p in E("LABELS", "data/simlabels_fut_hasty.pkl").split(","):
        if not os.path.exists(p): print("missing", p, flush=True); continue
        d = pickle.load(open(p, "rb")); labels += [x for x in d if x.get("fut") is not None]; print(f"{p}: {len(d)} positions", flush=True)
    rs = random.Random(0); rs.shuffle(labels); ntest = max(min(200, len(labels) // 5), len(labels) // 25); test, train = labels[:ntest], labels[ntest:]
    er = random.Random(1); EV = []
    for x in test:
        for ci, side, rows in units_of(x):
            if er.random() < 0.15:
                it = build_unit(x, ci, side, rows)
                if it is not None: EV.append(it)
        if len(EV) >= int(E("NEV", 1500)): break
    print(f"train positions {len(train)} test positions {ntest}; eval units {len(EV)} (me {sum(1 for e in EV if e['side']==0)}, opp {sum(1 for e in EV if e['side']==1)}); median mask {np.median([e['masksize'] for e in EV]):.0f}", flush=True)
    model = LatticeLex(R=R).to(dev)
    if E("RETR"):
        sd = torch.load(E("RETR"), map_location=dev); ms = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("wenc.slot") and k not in ("wenc.POS", "wenc.LEN")}, strict=False)
        print("init from", E("RETR"), "missing:", [m for m in ms.missing_keys if not m.startswith("refine") and not m.startswith("rnorm")])
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M R={R}", flush=True)

    @torch.no_grad()
    def evaluate(tag):
        model.eval(); keys = torch.cat([model.wenc(torch.arange(i, min(i + 8192, lex.NW), device=dev)).float() for i in range(0, lex.NW, 8192)])
        res = {s: {k: [] for k in ("m8", "m32", "ndcg", "pct")} for s in (0, 1)}; base = {s: {k: [] for k in ("m8", "m32", "ndcg", "pct")} for s in (0, 1)}
        for i in range(0, len(EV), 32):
            chunk = EV[i:i + 32]; b = {k: v.to(dev) for k, v in collate(chunk).items()}
            outs = model(b, keys=keys); key = model.rank_key(outs).float().cpu().numpy()
            for j, it in enumerate(chunk):
                n = len(it["cand"]); m = metrics(key[j, :n], it); mb = metrics(it["score"] + np.random.rand(n) * 1e-3, it)
                for k2 in m: res[it["side"]][k2].append(m[k2]); base[it["side"]][k2].append(mb[k2])
        model.train()
        for s, name in ((0, "me "), (1, "opp")):
            r_ = {k: np.nanmean(v) for k, v in res[s].items()}; b_ = {k: np.nanmean(v) for k, v in base[s].items()}
            print(f"  [{tag}] {name}: α-mass@8 {r_['m8']:.3f} @32 {r_['m32']:.3f} NDCG@32 {r_['ndcg']:.3f} pct {r_['pct']:.3f} | score-rank baseline @8 {b_['m8']:.3f} @32 {b_['m32']:.3f} NDCG {b_['ndcg']:.3f} pct {b_['pct']:.3f}", flush=True)
        return res
    dl = DataLoader(FDS(train, seed=0), batch_size=B, num_workers=NW_, collate_fn=collate, persistent_workers=NW_ > 0, prefetch_factor=4 if NW_ > 0 else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    evaluate("init"); t0 = time.time(); step = 0; la = None
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): outs = model(b)
        loss = 0.0
        for lg, al in outs:     # deep supervision: listwise CE against α restricted to the alive set
            t = b["tgt"] * al; tot = t.sum(1); m = tot > 0
            if m.any(): loss = loss + (-(t[m] / tot[m, None]) * F.log_softmax(lg[m].float(), 1)).sum(1).mean()
        loss = loss / len(outs)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
        if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.3f}", flush=True)
        if step % int(E("EVAL_EVERY", 2000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/future_{TAG}.pt")
        if step >= STEPS: break
    pickle.dump(evaluate("final"), open(f"data/result_future_{TAG}.pkl", "wb"))
