"""Step 2: fit a value net to sim labels. State = position AFTER a candidate move (board', leave, unseen, scores'); target =
sim estimate minus the move's own score (the future value of the leave on that board), in points/100.
Variants: USE_LEX=0 (position encoder only), USE_LEX=1 (rack-level + per-lane lexicon queries). Baseline: Macondo equity
(score + leave) affinely calibrated to the sim scale. Reports MAE in points on held-out positions.
Env: LABELS (pickle), USE_LEX, RETR, STEPS, B, LR, TAG, NLANE (lane queries), NWORK"""
import os, sys, time, math, pickle, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.data import lane_tokens, LANE_F, MAXL; from lexmem.policy import PolicyNet, retrieval_mask, RET_LENS, KRET
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get; USE_LEX = E("USE_LEX", "0") == "1"; STEPS = int(E("STEPS", 8000)); B = int(E("B", 64)); LR = float(E("LR", 3e-4))
TAG = E("TAG", "val"); NLANE = int(E("NLANE", 6)); NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))

def post_state(lex, x, ci):
    """apply candidate ci exactly -> (board', blanks', leave, unseen, scores', bag, turn+1, target)"""
    s, r, c, d, L, widx = x["cands"][ci][:6]
    b2, bl2, leave, placed = apply_move(lex, x["board"], x["blanks"], x["rack"], r, c, d, L, widx)
    target = (x["est"][ci] - s) / 100.0
    return b2, bl2, leave, x["unseen"], (x["scores"][0] + s, x["scores"][1]), x["bag"], x["turn"] + 1, target, x["cands"][ci][7] - s

def hot_lanes(lex, board, blanks, rack, unseen, n):
    """the n lanes with the most reachable words (letters on rack+unseen, <=3 missing); returns [(lane_tokens, L, mask)]"""
    cc = cross_checks(lex, board, blanks); out = []
    for (r, c, d, L) in lanes(board):
        if L < 4: continue
        info = lane_info(board, blanks, cc, r, c, d, L); g, loc = lane_candidates(lex, info, L)
        if len(loc) < 2: continue
        need = need_counts(lex, info, L, loc); rb = int(rack[26])
        deficit = np.maximum(need - rack[:26][None], 0); ok = (np.maximum(deficit - unseen[:26][None], 0).sum(1) <= rb) & (deficit.sum(1) - rb <= 3)
        if ok.sum() < 2: continue
        out.append((int(ok.sum()), lane_tokens(info, L), L, g[ok]))
    out.sort(key=lambda t: -t[0]); return [(t[1], t[2], t[3]) for t in out[:n]]

class VDS(Dataset):
    def __init__(self, items, n=10**7, seed=0, fixed=False):
        self.items, self.n, self.seed, self.rng, self.fixed = items, n, seed, None, fixed
        self.pairs = [(i, ci) for i, x in enumerate(items) for ci in range(len(x["cands"]))]
    def __len__(self): return len(self.pairs) if self.fixed else self.n
    def __getitem__(self, k):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + k)
        i, ci = self.pairs[k] if self.fixed else self.pairs[self.rng.randrange(len(self.pairs))]
        x = self.items[i]; b2, bl2, leave, unseen, sc, bag, turn, tgt, base = post_state(lex, x, ci)
        it = dict(board=b2.astype(np.int64) + 1, rack=leave.astype(np.float32), unseen=unseen.astype(np.float32),
                  glob=np.array([bag / 100, (sc[0] - sc[1]) / 100, turn / 30], dtype=np.float32), target=np.float32(tgt), base=np.float32(base))
        if USE_LEX:
            it["rmask"] = [retrieval_mask(lex, leave, unseen, L) for L in RET_LENS]
            it["lanes"] = hot_lanes(lex, b2, bl2, leave, unseen, NLANE)
        return it

def collate(items):
    B_ = len(items); out = {k: torch.tensor(np.stack([it[k] for it in items])) for k in ("board", "rack", "unseen", "glob")}
    out["target"] = torch.tensor([it["target"] for it in items]); out["base"] = torch.tensor([it["base"] for it in items])
    if USE_LEX:
        for li in range(len(RET_LENS)):
            C = max(len(it["rmask"][li]) for it in items); rm = torch.zeros(B_, C, dtype=torch.long); rv = torch.zeros(B_, C, dtype=torch.bool)
            for b, it in enumerate(items): n = len(it["rmask"][li]); rm[b, :n] = torch.tensor(it["rmask"][li]); rv[b, :n] = True
            out[f"rmask{li}"] = rm; out[f"rvalid{li}"] = rv
        lt = torch.zeros(B_, NLANE, MAXL, LANE_F); lL = torch.zeros(B_, NLANE, dtype=torch.long); C = max(max([len(m) for _, _, m in it["lanes"]] + [1]) for it in items)
        lm = torch.zeros(B_, NLANE, C, dtype=torch.long); lv = torch.zeros(B_, NLANE, C, dtype=torch.bool)
        for b, it in enumerate(items):
            for j, (tok, L, mask) in enumerate(it["lanes"]):
                lt[b, j] = torch.tensor(tok); lL[b, j] = L; lm[b, j, :len(mask)] = torch.tensor(mask); lv[b, j, :len(mask)] = True
        out.update(lane_tok=lt, lane_L=lL, lane_mask=lm, lane_valid=lv)
    return out

class ValueNet(PolicyNet):
    """PolicyNet's position encoder (+ lexicon branch) with extra per-lane queries; value-only forward."""
    def __init__(self, lex, use_lex, layers=None):
        super().__init__(lex, use_lex=use_lex, layers=layers or int(os.environ.get("VLAYERS", 3)))
        if use_lex: self.lane_type = nn.Parameter(torch.randn(1, 1, self.D) * 0.02)
    def forward(self, b):
        B_ = b["rack"].shape[0]; dev_ = b["rack"].device
        board = self.bemb(b["board"]).permute(0, 3, 1, 2); board = torch.cat([board, self.prem.expand(B_, -1, -1, -1)], 1)
        pos = torch.stack([self.cnn(board) + self.ttype[0], self.rack(b["rack"] / 3) + self.ttype[1], self.unseen(b["unseen"] / 12) + self.ttype[2], self.glob(b["glob"]) + self.ttype[3]], 1)
        toks = [self.cls.expand(B_, -1, -1), pos]; pads = [torch.zeros(B_, 5, dtype=torch.bool, device=dev_)]
        if self.use_lex:
            for li, L in enumerate(RET_LENS):
                lane = getattr(self, f"lane{L}").expand(B_, -1, -1)
                glob5 = torch.cat([b["glob"][:, :2], torch.full((B_, 1), L / 15, device=dev_), torch.zeros(B_, 2, device=dev_)], 1)
                q = self.qenc(b["rack"], b["unseen"], glob5, lane, b["board"], torch.full((B_,), L, device=dev_, dtype=torch.long))
                rm, rv = b[f"rmask{li}"], b[f"rvalid{li}"]
                logits = torch.einsum("bd,bcd->bc", q, self.keys[rm]) * (self.D ** -0.5); logits = logits.masked_fill(~rv, -1e4)
                k = min(KRET, rm.shape[1]); top = logits.topk(k, 1); idx = rm.gather(1, top.indices)
                need = (self.Wt[idx] - b["rack"][:, None, :26]).clamp_min(0).sum(-1)
                toks.append(self.keys[idx] + self.rfeat(torch.stack([top.values, need / 3], -1)) + self.ttype[5]); pads.append(~rv.gather(1, top.indices))
            # per-lane queries on the hottest lanes of the post-move board
            for j in range(b["lane_tok"].shape[1]):
                L = b["lane_L"][:, j]; valid_lane = L > 0
                glob5 = torch.cat([b["glob"][:, :2], (L.float() / 15)[:, None], torch.zeros(B_, 2, device=dev_)], 1)
                q = self.qenc(b["rack"], b["unseen"], glob5, b["lane_tok"][:, j], b["board"], L.clamp_min(1))
                rm, rv = b["lane_mask"][:, j], b["lane_valid"][:, j] & valid_lane[:, None]
                logits = torch.einsum("bd,bcd->bc", q, self.keys[rm]) * (self.D ** -0.5); logits = logits.masked_fill(~rv, -1e4)
                k = min(16, rm.shape[1]); top = logits.topk(k, 1); idx = rm.gather(1, top.indices)
                need = (self.Wt[idx] - b["rack"][:, None, :26]).clamp_min(0).sum(-1)
                toks.append(self.keys[idx] + self.rfeat(torch.stack([top.values, need / 3], -1)) + self.lane_type); pads.append(~rv.gather(1, top.indices))
        x = torch.cat(toks, 1); pad = torch.cat(pads, 1)
        h = self.norm(self.tr(x, src_key_padding_mask=pad)); return self.val(h[:, 0]).squeeze(-1)

if __name__ == "__main__":
    labels = pickle.load(open(E("LABELS", "data/simlabels_hasty300k.pkl"), "rb"))
    rs = random.Random(0); rs.shuffle(labels); ntest = max(200, len(labels) // 20); test, train = labels[:ntest], labels[ntest:]
    print(f"labels {len(labels)} positions, {sum(len(x['cands']) for x in labels)} (position, candidate) pairs; test {ntest}", flush=True)
    model = ValueNet(lex, USE_LEX).to(dev)
    if USE_LEX and E("RETR"): model.load_retrieval(E("RETR")); print("retrieval init from", E("RETR"))
    elif USE_LEX: model.refresh_keys()
    if E("INIT"): model.load_state_dict(torch.load(E("INIT"), map_location=dev), strict=False); print("init from", E("INIT"))
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M use_lex={USE_LEX} lanes={NLANE}", flush=True)
    dl = DataLoader(VDS(train, seed=0), batch_size=B, num_workers=NW_, collate_fn=collate, persistent_workers=NW_ > 0, prefetch_factor=4 if NW_ > 0 else None)
    tds = VDS(test, fixed=True); tdl = DataLoader(tds, batch_size=64, num_workers=min(NW_, 8), collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))

    @torch.no_grad()
    def evaluate(tag):
        model.eval(); pred, tgt, base = [], [], []
        for b in tdl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v = model(b)
            pred += v.float().cpu().tolist(); tgt += b["target"].cpu().tolist(); base += b["base"].cpu().tolist()
        pred, tgt, base = np.array(pred) * 100, np.array(tgt) * 100, np.array(base)
        A = np.vstack([base, np.ones_like(base)]).T; coef = np.linalg.lstsq(A, tgt, rcond=None)[0]; base_cal = A @ coef
        model.train()
        print(f"  [{tag}] MAE net={np.abs(pred-tgt).mean():.2f} pts | baseline (leave table, affine-calibrated) {np.abs(base_cal-tgt).mean():.2f} | predict-mean {np.abs(tgt-tgt.mean()).mean():.2f} | corr net {np.corrcoef(pred,tgt)[0,1]:.3f} baseline {np.corrcoef(base,tgt)[0,1]:.3f}", flush=True)
        return float(np.abs(pred - tgt).mean())

    evaluate("untrained"); t0 = time.time(); step = 0; la = None
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): v = model(b)
        loss = F.smooth_l1_loss(v.float(), b["target"])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; la = loss.item() if la is None else 0.98 * la + 0.02 * loss.item()
        if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss={la:.4f}", flush=True)
        if step % int(E("EVAL_EVERY", 1000)) == 0 or step == STEPS: evaluate(f"step {step}"); torch.save(model.state_dict(), f"data/value_{TAG}.pt")
        if step >= STEPS: break
    mae = evaluate("final"); pickle.dump(dict(mae=mae, use_lex=USE_LEX), open(f"data/result_value_{TAG}.pkl", "wb"))
