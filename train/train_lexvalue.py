"""Stage 2: does explicit memory over the whole lexicon make a better value function? Lex vs no-lex, everything else equal.
Unit of data: the afterstate of a played move (board after the move, the mover's leave, unseen tiles from the mover's view, scores),
from Go self-play games (prep/go2pkl.py full conversion). Targets come only from the games themselves:
  value: final spread - spread after the move (/100, mover's view) and win (1 / 0.5 / 0);  aux (lex arm): the opponent's actual next word.
Lanes: NLANES open lanes of the afterstate chosen by a lexicon-free heuristic (word mult x letter mults over the empty squares),
identical for both arms. Per lane and side (mover, opponent) one token:
  ARM=lex   : QueryEncoder(lane, side's tiles) -> retrieval over the side's exact rules mask in the full lexicon (frozen word keys)
              -> token from the query, the softmax-pooled retrieved keys, best / expected lane score, log mask size;
  ARM=nolex : the same QueryEncoder query alone (no candidates, no word keys, no scores).
Trunk: board CNN + leave + unseen + globals + the lane tokens -> 3-layer transformer -> value (spread change, win logit).
Held out: games with gi % HOLD == 0; metrics by phase: spread MAE (points), win log-loss; lex: opponent next-word top-8 in its lane.
Usage: ARM=lex TRAIN=spf_0,spf_1 INIT_FUT=data/future_R1.pt STEPS=40000 TAG=lexvalue_lex python train/train_lexvalue.py"""
import os, time, math, pickle, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.data import lane_tokens, LANE_F, MAXL; from lexmem.model import WordEncoder, QueryEncoder
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
ARM = E("ARM", "lex"); LEXARM = ARM == "lex"; NL = int(E("NLANES", 12)); CAP = int(E("CAP", 256)); STEPS = int(E("STEPS", 40000))
B = int(E("B", 64)); LR = float(E("LR", 2e-4)); HOLD = int(E("HOLDOUT_MOD", 20)); NW = int(E("NWORK", 24)); TAG = E("TAG", f"lexvalue_{ARM}")
AUXW = float(E("AUX_W", 0.5)); NEV = int(E("NEV", 4000)); NEV_SUB = int(E("NEV_SUB", 4000)); D = 256
TARGET = E("TARGET", "outcome")   # outcome: the game's own result; ro: mean of K rollouts from the afterstate (rec["ro"])
FREEZE_Q = E("FREEZE_Q", "0") == "1"; MAX_TRAIN = int(E("MAX_TRAIN", 0))   # frozen query encoder (both arms); fixed training subsample
SEED = int(E("SEED", 0)); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))

def pick_lanes(b2):
    """lexicon-free: top NL lanes by word-mult product x letter-mult sum over empty squares (+ length), one per start square."""
    best = {}
    for (r, c, d, L) in lanes(b2):
        dr, dc = ((0, 1), (1, 0))[d]; rows = r + dr * np.arange(L); cols = c + dc * np.arange(L); em = b2[rows, cols] < 0
        h = float(np.prod(np.where(em, WMULT[rows, cols], 1)) * (LMULT[rows, cols][em].sum()) + 0.1 * L)
        if (r, c, d) not in best or h > best[(r, c, d)][0]: best[(r, c, d)] = (h, L)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:NL]
    return [(r, c, d, L) for (r, c, d), (h, L) in top]

def side_mask(info, L, loc, leave, unseen, side):
    need = need_counts(lex, info, L, loc)
    if side == 0:
        deficit = np.maximum(need - leave[:26][None], 0)
        return (np.maximum(deficit - unseen[:26][None], 0).sum(1) <= int(leave[26]) + int(unseen[26])) & (deficit.sum(1) - int(leave[26]) <= 3)
    return np.maximum(need - unseen[:26][None], 0).sum(1) <= int(unseen[26])

def build(rec, final_scores, nxt):
    """afterstate item for the move played at rec (a play), or None."""
    m = rec["move"]
    if m[0] != "play": return None
    _, r0, c0, d0, L0, letters, s = m
    w0 = lex.WIDX.get("".join(LET[rec["board"][r0 + ((0, 1), (1, 0))[d0][0] * j, c0 + ((0, 1), (1, 0))[d0][1] * j]] if ch == "." else ch.upper() for j, ch in enumerate(letters)))
    if w0 is None: return None
    b2, bl2, leave, _ = apply_move(lex, rec["board"], rec["blanks"], rec["rack"], r0, c0, d0, L0, w0)
    unseen = rec["unseen"]; me = rec["mover"]; sp_after = rec["scores"][0] + s - rec["scores"][1]
    if TARGET == "ro":
        if "ro" not in rec: return None
        G = np.float32((rec["ro"][0] - sp_after) / 100); W = np.float32(rec["ro"][1])
    else:
        fin = (final_scores[me] - final_scores[1 - me]); G = np.float32((fin - sp_after) / 100); W = np.float32(1.0 if fin > 0 else 0.5 if fin == 0 else 0.0)
    cc = cross_checks(lex, b2, bl2); ln = pick_lanes(b2)
    if not ln: return None
    aux_key = None
    if nxt is None and "next_move" in rec: nxt = dict(move=rec["next_move"])
    if nxt is not None and nxt["move"][0] == "play":   # the opponent's actual next play, as (lane, word)
        _, r1, c1, d1, L1, let1, _ = nxt["move"]; dr1, dc1 = ((0, 1), (1, 0))[d1]
        aux_key = ((r1, c1, d1, L1), lex.WIDX.get("".join(LET[b2[r1 + dr1 * j, c1 + dc1 * j]] if ch == "." else ch.upper() for j, ch in enumerate(let1))))
    U = []
    for (r, c, d, L) in ln:
        info = lane_info(b2, bl2, cc, r, c, d, L); lt = lane_tokens(info, L)
        for side in (0, 1):
            u = dict(lane=lt, L=L, side=side, glob=np.array([len_bag(rec) / 100, sp_after / 100 * (1 if side == 0 else -1), L / 15, 0.0, float(side)], dtype=np.float32))
            if LEXARM:
                g, loc = lane_candidates(lex, info, L)
                if len(loc): ok = side_mask(info, L, loc, leave, unseen, side); g, loc = g[ok], loc[ok]
                sc = lane_scores(lex, info, L, loc).astype(np.float32) if len(loc) else np.zeros(0, np.float32)
                if len(g) > CAP: keep = np.argsort(-sc)[:CAP]; g, sc = g[keep], sc[keep]
                a = -1
                if aux_key is not None and side == 1 and aux_key[0] == (r, c, d, L) and aux_key[1] is not None:
                    hit = np.where(g == aux_key[1])[0]; a = int(hit[0]) if len(hit) else -1
                u.update(cand=g.astype(np.int64), score=sc, msize=np.float32(math.log1p(len(loc))), aux=a)
            U.append(u)
    return dict(board=b2.astype(np.int64) + 1, leave=leave.astype(np.float32), unseen=unseen.astype(np.float32),
                glob=np.array([len_bag(rec) / 100, sp_after / 100, rec["turn"] / 30], dtype=np.float32), units=U, G=G, W=W, bag=rec["bag"])
def len_bag(rec): return max(0, int(rec["unseen"].sum()) - 7)

def collate(items):
    Bn = len(items); Un = NL * 2
    out = {k: torch.tensor(np.stack([it[k] for it in items])) for k in ("board", "leave", "unseen", "glob")}
    out["G"] = torch.tensor([it["G"] for it in items]); out["W"] = torch.tensor([it["W"] for it in items]); out["bag"] = torch.tensor([it["bag"] for it in items])
    lane = torch.zeros(Bn, Un, MAXL, LANE_F); Lv = torch.ones(Bn, Un, dtype=torch.long) * 2; g5 = torch.zeros(Bn, Un, 5); uval = torch.zeros(Bn, Un, dtype=torch.bool)
    for b, it in enumerate(items):
        for j, u in enumerate(it["units"]):
            lane[b, j] = torch.tensor(u["lane"]); Lv[b, j] = u["L"]; g5[b, j] = torch.tensor(u["glob"]); uval[b, j] = True
    out.update(lane=lane, L=Lv, g5=g5, uvalid=uval)
    if LEXARM:
        C = max(1, max(len(u["cand"]) for it in items for u in it["units"]))
        cand = torch.zeros(Bn, Un, C, dtype=torch.long); cv = torch.zeros(Bn, Un, C, dtype=torch.bool); sc = torch.zeros(Bn, Un, C); ms = torch.zeros(Bn, Un); aux = torch.full((Bn, Un), -1, dtype=torch.long)
        for b, it in enumerate(items):
            for j, u in enumerate(it["units"]):
                n = len(u["cand"])
                if n: cand[b, j, :n] = torch.from_numpy(u["cand"]); cv[b, j, :n] = True; sc[b, j, :n] = torch.from_numpy(u["score"])
                ms[b, j] = float(u["msize"]); aux[b, j] = u["aux"]
        out.update(cand=cand, cvalid=cv, cscore=sc, msize=ms, aux=aux)
    return out

class LexValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.qenc = QueryEncoder(D, 3); self.bemb = nn.Embedding(27, 32)
        self.cnn = nn.Sequential(nn.Conv2d(32 + 5, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 16, 1), nn.GELU(), nn.Flatten(), nn.Linear(16 * 225, D))
        self.register_buffer("prem", F.one_hot(torch.tensor(PREM), 5).permute(2, 0, 1).float()[None])
        self.leave = nn.Linear(27, D); self.unseen = nn.Linear(27, D); self.glob = nn.Linear(3, D)
        self.ttype = nn.Parameter(torch.randn(6, D) * 0.02); self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.uq = nn.Linear(D, D)
        if LEXARM:
            self.wenc = WordEncoder(lex, D, slot=False); self.register_buffer("keys", torch.zeros(lex.NW, D))
            self.up = nn.Linear(D, D); self.us = nn.Linear(3, D); self.scale = nn.Parameter(torch.tensor(D ** -0.5))
        enc = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 3); self.norm = nn.LayerNorm(D); self.val = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))
    @torch.no_grad()
    def refresh_keys(self):
        for i in range(0, lex.NW, 8192):
            idx = torch.arange(i, min(i + 8192, lex.NW), device=self.keys.device); self.keys[i:i + 8192] = self.wenc(idx).float()
    def forward(self, b):
        Bn, Un = b["lane"].shape[:2]
        board = torch.cat([self.bemb(b["board"]).permute(0, 3, 1, 2), self.prem.expand(Bn, -1, -1, -1)], 1)
        q = self.qenc(b["leave"].repeat_interleave(Un, 0), b["unseen"].repeat_interleave(Un, 0), b["g5"].flatten(0, 1), b["lane"].flatten(0, 1),
                      b["board"].repeat_interleave(Un, 0), b["L"].flatten())          # (Bn*Un, D)
        tok = self.uq(q); aux_logits = None
        if LEXARM:
            k = self.keys[b["cand"].flatten(0, 1)]; cv = b["cvalid"].flatten(0, 1)                 # (N, C, D)
            lg = torch.einsum("nd,ncd->nc", q, k) * self.scale; lg = lg.masked_fill(~cv, -1e4); aux_logits = lg
            p = torch.softmax(lg, 1) * cv.any(1, keepdim=True); pooled = torch.einsum("nc,ncd->nd", p, k)
            sc = b["cscore"].flatten(0, 1); best = (sc * cv).amax(1) / 50; expd = (p * sc).sum(1) / 50
            tok = tok + self.up(pooled) + self.us(torch.stack([best, expd, b["msize"].flatten()], 1))
        tok = tok.view(Bn, Un, D) + self.ttype[4 + 0]
        pos = torch.stack([self.cnn(board) + self.ttype[0], self.leave(b["leave"] / 3) + self.ttype[1], self.unseen(b["unseen"] / 12) + self.ttype[2], self.glob(b["glob"]) + self.ttype[3]], 1)
        x = torch.cat([self.cls.expand(Bn, -1, -1), pos, tok], 1)
        pad = torch.cat([torch.zeros(Bn, 5, dtype=torch.bool, device=x.device), ~b["uvalid"]], 1)
        h = self.norm(self.tr(x, src_key_padding_mask=pad))
        return self.val(h[:, 0]), aux_logits

class DS(Dataset):
    def __init__(self, keys, seed): self.keys, self.rng = keys, random.Random(seed)
    def __len__(self): return 10 ** 9
    def __getitem__(self, i):
        while True:
            gi, ti = self.keys[self.rng.randrange(len(self.keys))]; tr, fs = games[gi]
            it = build(tr[ti], fs, None if "next_move" in tr[ti] else (tr[ti + 1] if ti + 1 < len(tr) else None))
            if it is not None: return it
class DSfixed(Dataset):
    def __init__(self, keys): self.keys = keys
    def __len__(self): return len(self.keys)
    def __getitem__(self, i):
        gi, ti = self.keys[i]; tr, fs = games[gi]
        return build(tr[ti], fs, None if "next_move" in tr[ti] else (tr[ti + 1] if ti + 1 < len(tr) else None))

if __name__ == "__main__":
    games = [g for t in E("TRAIN").split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]
    keys_all = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti, r in enumerate(tr) if r["move"][0] == "play" and (TARGET != "ro" or "ro" in r)]
    train = [k for k in keys_all if k[0] % HOLD != 0]; held = [k for k in keys_all if k[0] % HOLD == 0]
    if MAX_TRAIN and MAX_TRAIN < len(train): train = random.Random(123).sample(train, MAX_TRAIN)   # same subsample for every arm and seed
    print(f"ARM={ARM}: {len(games)} games, {len(train)} train / {len(held)} held-out afterstates", flush=True)
    model = LexValue().to(dev)
    if E("INIT_FUT"):
        sd = torch.load(E("INIT_FUT"), map_location=dev)
        model.qenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("qenc.")})
        if LEXARM:
            model.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.") and not k.startswith("wenc.slot") and k not in ("wenc.POS", "wenc.LEN")}, strict=False)
            model.refresh_keys()
            for p in model.wenc.parameters(): p.requires_grad_(False)   # word keys frozen: the memory, not a trainable embedding table
        print("query encoder (and word encoder) from", E("INIT_FUT"), flush=True)
    if FREEZE_Q:
        for p in model.qenc.parameters(): p.requires_grad_(False)
    print(f"freeze_q={FREEZE_Q} max_train={MAX_TRAIN or 'all'} seed={SEED}", flush=True)
    print(f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)
    rs = random.Random(7); ev_keys = rs.sample(held, min(NEV, len(held)))
    EV = [x for chunk in DataLoader(DSfixed(ev_keys), batch_size=256, num_workers=NW, collate_fn=lambda b: b) for x in chunk if x is not None]
    @torch.no_grad()
    def evaluate(tag, full=False):
        model.eval(); rows = []; ev = EV if full else EV[:NEV_SUB]
        for i in range(0, len(ev), 64):
            b = {k: v.to(dev) for k, v in collate(ev[i:i + 64]).items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, al = model(b)
            v = v.float(); ll = F.binary_cross_entropy_with_logits(v[:, 1], b["W"], reduction="none")
            for j in range(len(b["G"])): rows.append((float((v[j, 0] - b["G"][j]).abs()) * 100, float(ll[j]), int(b["bag"][j]), float(v[j, 0] - b["G"][j]) ** 2, float(b["G"][j])))
            if LEXARM:
                a = b["aux"].flatten(); m = a >= 0
                if m.any():
                    top = al.float()[m].topk(min(8, al.shape[1]), 1).indices; rows_aux.append(((top == a[m][:, None]).any(1).float().cpu().numpy()))
        model.train(); r = np.array(rows)
        def r2(k): return 1 - r[k, 3].mean() / max(r[k, 4].var(), 1e-9)
        allk = np.ones(len(r), dtype=bool)
        line = f"[{tag}] held-out n={len(r)}: spread R2 {r2(allk):.3f} MAE {r[:,0].mean():.2f} pts, win log-loss {r[:,1].mean():.4f}"
        for lo, hi, nm in ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (0, 7, "pre-end/end")):
            k = (r[:, 2] >= lo) & (r[:, 2] <= hi)
            if k.sum(): line += f" | {nm}: R2 {r2(k):.3f}, LL {r[k,1].mean():.3f}"
        if LEXARM and rows_aux: line += f" | opp next word in top-8 of its lane: {np.concatenate(rows_aux).mean():.3f}"
        rows_aux.clear(); print(line, flush=True)
    rows_aux = []
    print(f"eval afterstates {len(EV)}", flush=True); evaluate("init")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    t0 = time.time(); step = 0; la = None
    for b in DataLoader(DS(train, 0), batch_size=B, num_workers=NW, collate_fn=collate, persistent_workers=True, prefetch_factor=4):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v, al = model(b)
        v = v.float(); loss = F.smooth_l1_loss(v[:, 0], b["G"]) + F.binary_cross_entropy_with_logits(v[:, 1], b["W"])
        if LEXARM:
            a = b["aux"].flatten(); m = a >= 0
            if m.any(): loss = loss + AUXW * F.cross_entropy(al.float()[m], a[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
        if step % 200 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}", flush=True)
        if step % int(E("EVAL_EVERY", 4000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/{TAG}.pt")
        if step >= STEPS: break
    evaluate("final, all eval afterstates", full=True); print(f"=== saved data/{TAG}.pt", flush=True)
