"""Dataset: (position, lane) -> query features + candidate set + reachable weights. Labels computed on the fly."""
import numpy as np, random, pickle, torch
from torch.utils.data import Dataset
from lexmem.core import *

LANE_F = 27 + 26 + 1 + 5 + 1        # fixed onehot, allowed, has-cross, premium onehot, is-empty
MAXL = 15

def lane_tokens(info, L):
    x = np.zeros((MAXL, LANE_F), dtype=np.float32)
    f = info["fixed"]
    for j in range(L):
        x[j, 26 if f[j] < 0 else f[j]] = 1.0
        x[j, 27:53] = info["allow"][j]
        x[j, 53] = info["has"][j]
        x[j, 54 + info["prem"][j]] = 1.0
        x[j, 59] = info["empty"][j]
    return x

def holdout_mask(lex, frac=0.05, seed=123):
    rs = np.random.RandomState(seed)
    return rs.rand(lex.NW) < frac

class LaneDataset(Dataset):
    """Infinite-style dataset: index -> random (game, turn, lane). Uses per-worker rng."""
    def __init__(self, lex, games, heldout, n_items=10**7, n_neg=512, n_pos=256, p_construct=0.5,
                 exclude_heldout=True, seed=0, miss_choices=(0, 1, 2, 3)):
        self.lex, self.games, self.heldout = lex, games, heldout
        self.n_items, self.n_neg, self.n_pos = n_items, n_neg, n_pos
        self.p_construct, self.exclude = p_construct, exclude_heldout
        self.seed = seed; self.rng = None; self.miss_choices = miss_choices
        self.positions = [(gi, ti) for gi, (traj, _) in enumerate(games) for ti in range(len(traj))]
        self.cache = {}

    def __len__(self): return self.n_items

    def _cc(self, gi, ti):
        key = (gi, ti)
        if key not in self.cache:
            if len(self.cache) > 64: self.cache.clear()
            rec = self.games[gi][0][ti]
            cc = cross_checks(self.lex, rec["board"], rec["blanks"])
            ll = lanes(rec["board"])
            self.cache[key] = (cc, ll)
        return self.cache[key]

    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info()
            self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + i)
        rng = self.rng
        for _ in range(50):
            gi, ti = self.positions[rng.randrange(len(self.positions))]
            rec = self.games[gi][0][ti]
            cc, ll = self._cc(gi, ti)
            if not ll: continue
            r, c, d, L = ll[rng.randrange(len(ll))]
            info = lane_info(rec["board"], rec["blanks"], cc, r, c, d, L)
            g, loc = lane_candidates(self.lex, info, L)
            if len(loc) < 2: continue
            rack, unseen = rec["rack"].copy(), rec["unseen"].copy()
            constructed = rng.random() < self.p_construct
            if constructed:
                # build the rack around a random candidate word, leaving `miss` of its letters in U
                k = rng.randrange(len(loc))
                need = need_counts(self.lex, info, L, loc[k:k+1])[0]
                pool = rack + unseen
                needl = [l for l in range(26) for _ in range(need[l])]
                if len(needl) > 7: continue
                m = min(rng.choice(self.miss_choices), len(needl))
                rng.shuffle(needl); keep = needl[m:]
                newrack = np.zeros(27, dtype=np.int16)
                okc = True
                for l in keep:
                    if pool[l] <= 0: okc = False; break
                    newrack[l] += 1; pool[l] -= 1
                if not okc: continue
                while newrack.sum() < 7 and pool.sum() > 0:
                    t = rng.choices(range(27), weights=pool.tolist())[0]
                    newrack[t] += 1; pool[t] -= 1
                rack, unseen = newrack, pool
            ok, miss, P, score, w = reachable_label(self.lex, info, L, loc, rack, unseen)
            if self.exclude:
                keepm = ~self.heldout[g]
            else:
                keepm = np.ones(len(g), dtype=bool)
            pos = np.where(ok & keepm)[0]; neg = np.where(~ok & keepm)[0]
            if len(pos) == 0: continue
            if len(pos) > self.n_pos: pos = np.array(rng.sample(pos.tolist(), self.n_pos))
            if len(neg) > self.n_neg: neg = np.array(rng.sample(neg.tolist(), self.n_neg))
            sel = np.concatenate([pos, neg])
            return dict(rack=rack.astype(np.float32), unseen=unseen.astype(np.float32),
                        glob=np.array([rec["bag"] / 100, (rec["scores"][0] - rec["scores"][1]) / 100, L / 15, len(loc) / 1000, float(constructed)], dtype=np.float32),
                        lane=lane_tokens(info, L), board=rec["board"].astype(np.int64) + 1,
                        cand=g[sel].astype(np.int64), weight=w[sel].astype(np.float32), ok=ok[sel],
                        miss=miss[sel].astype(np.int64), P=P[sel].astype(np.float32), score=score[sel].astype(np.float32),
                        masksize=len(loc), n_ok=int(ok.sum()), L=L)
        raise RuntimeError("could not construct an item")

def collate(items):
    B = len(items); C = max(len(it["cand"]) for it in items)
    out = {}
    for k in ("rack", "unseen", "glob", "lane", "board"):
        out[k] = torch.tensor(np.stack([it[k] for it in items]))
    cand = torch.zeros(B, C, dtype=torch.long); valid = torch.zeros(B, C, dtype=torch.bool)
    weight = torch.zeros(B, C); ok = torch.zeros(B, C, dtype=torch.bool); miss = torch.full((B, C), -1, dtype=torch.long)
    P = torch.zeros(B, C); score = torch.zeros(B, C)
    for b, it in enumerate(items):
        n = len(it["cand"]); cand[b, :n] = torch.tensor(it["cand"]); valid[b, :n] = True
        weight[b, :n] = torch.tensor(it["weight"]); ok[b, :n] = torch.tensor(it["ok"]); miss[b, :n] = torch.tensor(it["miss"])
        P[b, :n] = torch.tensor(it["P"]); score[b, :n] = torch.tensor(it["score"])
    out.update(cand=cand, valid=valid, weight=weight, ok=ok, miss=miss, P=P, score=score,
               masksize=torch.tensor([it["masksize"] for it in items]), n_ok=torch.tensor([it["n_ok"] for it in items]),
               L=torch.tensor([it["L"] for it in items]))
    return out
