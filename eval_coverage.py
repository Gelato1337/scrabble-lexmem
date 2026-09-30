"""
Coverage sweep over the ENTIRE lexicon.
For every word w and every m in {0,1,2}: find a real test-board lane where w fits (pattern + cross-checks), build the
mover's rack from w's needed letters minus m (left in the bag) plus random filler, compute the exact rules mask and
reachable label, and ask the trained model:
   masked   : rank of w inside its rules mask (percentile; top-32 hit)
   unmasked : rank of w among ALL 168k words for the same query (no rules mask at all)
Reports by word length and m, and the worst-retrieved words.
Env: CKPT (checkpoint), SLOT (1/0 must match the checkpoint), TEST (games file tag), NWORK, OUT
"""
import os, sys, time, pickle, random, numpy as np, torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.data import *; from lexmem.model import *
import warnings; warnings.filterwarnings("ignore")
CKPT = os.environ.get("CKPT", "data/lexmem_base.pt"); SLOT = os.environ.get("SLOT", "1") == "1"
NW_ = int(os.environ.get("NWORK", os.environ.get("SLURM_CPUS_PER_TASK", 8))); OUT = os.environ.get("OUT", "data/coverage.pkl")
LIMIT = int(os.environ.get("LIMIT", 0)); dev = "cuda" if torch.cuda.is_available() else "cpu"
lex = Lexicon(os.environ.get("LEXICON", os.path.join(ROOT, "enable1.txt"))); heldout = holdout_mask(lex)
games = pickle.load(open(f"data/games_{os.environ.get('TEST','test')}.pkl", "rb"))
MS = (0, 1, 2)

# lane pool from test positions, grouped by length; each entry: (gi, ti, r, c, d)
positions = [(gi, ti) for gi, (traj, _) in enumerate(games) for ti in range(len(traj))]
rs = random.Random(0); rs.shuffle(positions); positions = positions[:600]
POOL0 = defaultdict(list); POOLF = defaultdict(lambda: defaultdict(list)); CC = {}
for gi, ti in positions:
    rec = games[gi][0][ti]; cc = cross_checks(lex, rec["board"], rec["blanks"]); CC[(gi, ti)] = cc
    for (r, c, d, L) in lanes(rec["board"]):
        dr, dc = ((0, 1), (1, 0))[d]; fixed = [(j, rec["board"][r + dr * j, c + dc * j]) for j in range(L)]
        fixed = [(j, l) for j, l in fixed if l >= 0]
        if not fixed: POOL0[L].append((gi, ti, r, c, d))
        else:
            for j, l in fixed: POOLF[L][(j, int(l))].append((gi, ti, r, c, d))
print("lane pool: no-fixed by length", {L: len(v) for L, v in sorted(POOL0.items())}, flush=True)
EMPTY = [(gi, ti) for gi, ti in positions if games[gi][0][ti]["turn"] == 0]
if not EMPTY:
    EMPTY = [(gi, 0) for gi in range(len(games))]
    for gi, ti in EMPTY:
        rec = games[gi][0][ti]; CC[(gi, ti)] = cross_checks(lex, rec["board"], rec["blanks"])
def lanes_for(wi, L, rng, n=None):
    n = n or (80 if L <= 7 else 400)
    out = []
    if L <= 7 and POOL0.get(L): out += [POOL0[L][rng.randrange(len(POOL0[L]))] for _ in range(n // 2)]
    js = list(range(L)); rng.shuffle(js)
    for j in js:
        lst = POOLF[L].get((j, int(lex.POS[wi, j])), [])
        if lst: out += [lst[rng.randrange(len(lst))] for _ in range(max(2, n // (2 * L)))]
    rng.shuffle(out); out = out[:n]
    if L <= 7:   # guaranteed fallback: empty board, lane through the centre
        for _ in range(6):
            gi, ti = EMPTY[rng.randrange(len(EMPTY))]; d = rng.randrange(2); off = rng.randrange(L)
            out.append((gi, ti, 7, 7 - off, 0) if d == 0 else (gi, ti, 7 - off, 7, 1))
    return out

class CovDS(Dataset):
    def __len__(self): return (LIMIT or lex.NW) * len(MS)
    def __getitem__(self, i):
        wi, mi = divmod(i, len(MS)); m = MS[mi]; w = lex.words[wi]; L = len(w); rng = random.Random(i)
        need_full = lex.W[wi]
        for (gi, ti, r, c, d) in lanes_for(wi, L, rng):
            rec = games[gi][0][ti]; info = lane_info(rec["board"], rec["blanks"], CC[(gi, ti)], r, c, d, L)
            f = info["fixed"]; al = info["allow"]
            if any((f[j] >= 0 and f[j] != lex.POS[wi, j]) or (f[j] < 0 and not al[j][lex.POS[wi, j]]) for j in range(L)): continue
            g, loc = lane_candidates(lex, info, L)
            k = np.where(g == wi)[0]
            if len(k) == 0: continue
            need = need_counts(lex, info, L, loc[k[0]:k[0] + 1])[0]
            needl = [l for l in range(26) for _ in range(need[l])]
            if len(needl) > 7 or len(needl) == 0: continue
            pool = rec["rack"] + rec["unseen"]
            mm = min(m, len(needl)); rng.shuffle(needl); keep, moved = needl[mm:], needl[:mm]
            newrack = np.zeros(27, dtype=np.int16); okc = True
            for l in needl:
                if pool[l] <= 0: okc = False; break
                pool[l] -= 1
            if not okc: continue
            for l in moved: pool[l] += 1          # moved letters stay in the bag
            for l in keep: newrack[l] += 1
            while newrack.sum() < 7 and pool.sum() > 0:
                t = rng.choices(range(27), weights=pool.tolist())[0]; newrack[t] += 1; pool[t] -= 1
            ok, miss, P, score, wt = reachable_label(lex, info, L, loc, newrack, pool)
            return dict(wi=wi, m=m, L=L, found=True, rack=newrack.astype(np.float32), unseen=pool.astype(np.float32),
                        glob=np.array([rec["bag"] / 100, (rec["scores"][0] - rec["scores"][1]) / 100, L / 15, len(loc) / 1000, 1.0], dtype=np.float32),
                        lane=lane_tokens(info, L), board=rec["board"].astype(np.int64) + 1, cand=g.astype(np.int64),
                        target_pos=int(k[0]), target_ok=bool(ok[k[0]]), masksize=len(loc), n_ok=int(ok.sum()))
        # synthetic fallback (long words): empty board with some of the word's own letters pre-placed in row 7
        for _ in range(8):
            gi, ti = positions[rng.randrange(len(positions))]; rec = games[gi][0][ti]
            board, blanks = empty_board(); c0 = (N - L) // 2
            nfix = max(1, L - 7) + rng.randrange(0, 3); js = sorted(rng.sample(range(L), min(nfix, L - 1)))
            for j in js: board[7, c0 + j] = lex.POS[wi, j]
            cc = cross_checks(lex, board, blanks); info = lane_info(board, blanks, cc, 7, c0, 0, L)
            g, loc = lane_candidates(lex, info, L); k = np.where(g == wi)[0]
            if len(k) == 0: continue
            need = need_counts(lex, info, L, loc[k[0]:k[0] + 1])[0]; needl = [l for l in range(26) for _ in range(need[l])]
            if len(needl) > 7 or len(needl) == 0: continue
            pool = rec["rack"] + rec["unseen"]
            mm = min(m, len(needl)); rng.shuffle(needl); keep, moved = needl[mm:], needl[:mm]
            newrack = np.zeros(27, dtype=np.int16); okc = True
            for l in needl:
                if pool[l] <= 0: okc = False; break
                pool[l] -= 1
            if not okc: continue
            for l in moved: pool[l] += 1
            for l in keep: newrack[l] += 1
            while newrack.sum() < 7 and pool.sum() > 0:
                t = rng.choices(range(27), weights=pool.tolist())[0]; newrack[t] += 1; pool[t] -= 1
            ok, miss, P, score, wt = reachable_label(lex, info, L, loc, newrack, pool)
            return dict(wi=wi, m=m, L=L, found=True, rack=newrack.astype(np.float32), unseen=pool.astype(np.float32),
                        glob=np.array([rec["bag"] / 100, 0.0, L / 15, len(loc) / 1000, 1.0], dtype=np.float32),
                        lane=lane_tokens(info, L), board=board.astype(np.int64) + 1, cand=g.astype(np.int64),
                        target_pos=int(k[0]), target_ok=bool(ok[k[0]]), masksize=len(loc), n_ok=int(ok.sum()))
        return dict(wi=wi, m=m, L=L, found=False)

def coll(items): return items
model = LexMem(lex, D=256, slot=SLOT, qlayers=3).to(dev); model.load_state_dict(torch.load(CKPT, map_location=dev)); model.eval()
with torch.no_grad():
    keys = torch.cat([model.wenc(torch.arange(i, min(i + 8192, lex.NW), device=dev)).float() for i in range(0, lex.NW, 8192)])
dl = DataLoader(CovDS(), batch_size=64, num_workers=NW_, collate_fn=coll)
rows = []; t0 = time.time(); n = 0
with torch.no_grad():
    for items in dl:
        good = [it for it in items if it["found"]]
        for it in items:
            if not it["found"]: rows.append((it["wi"], it["m"], it["L"], False, np.nan, np.nan, np.nan, 0, 0, False))
        if good:
            b = {k: torch.tensor(np.stack([it[k] for it in good])).to(dev) for k in ("rack", "unseen", "glob", "lane", "board")}
            b["L"] = torch.tensor([it["L"] for it in good], device=dev)
            q = model.qenc(b["rack"], b["unseen"], b["glob"], b["lane"], b["board"], b["L"]).float()
            full = (q @ keys.T) * model.scale                          # (B, NW) unmasked
            for bi, it in enumerate(good):
                s_full = full[bi]; cand = torch.tensor(it["cand"], device=dev)
                s_mask = s_full[cand]; tgt = it["target_pos"]
                rank_m = int((s_mask > s_mask[tgt]).sum().item()); pct_m = rank_m / max(len(cand) - 1, 1)
                rank_f = int((s_full > s_full[it["wi"]]).sum().item())
                rows.append((it["wi"], it["m"], it["L"], True, pct_m, rank_m, rank_f, it["masksize"], it["n_ok"], it["target_ok"]))
        n += len(items)
        if n % (64 * 200) == 0: print(f"{n}/{len(dl.dataset)} items {time.time()-t0:.0f}s", flush=True)
import numpy as np
R = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]) for r in rows], dtype=object)
pickle.dump(rows, open(OUT, "wb"))
wi = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows]); found = np.array([r[3] for r in rows])
pct = np.array([r[4] for r in rows], dtype=float); rank_m = np.array([r[5] for r in rows], dtype=float); rank_f = np.array([r[6] for r in rows], dtype=float)
msz = np.array([r[7] for r in rows], dtype=float); tok = np.array([r[9] for r in rows])
print(f"\nwords {len(set(wi.tolist()))}  items {len(rows)}  constructible {found.mean():.3f}  target-in-reachable-set {tok[found].mean():.3f}")
print("\nBy missing letters m (constructible items):")
print(f"{'m':>2} {'n':>7} {'pct(mask)':>10} {'top32(mask)':>12} {'top1(mask)':>11} {'unmasked median rank':>21} {'unmasked top32':>15} {'unmasked top256':>16} {'heldout pct':>12}")
for mm in MS:
    s = found & (m == mm)
    ho = s & heldout[wi]
    print(f"{mm:>2} {s.sum():>7} {np.nanmean(pct[s]):>10.3f} {np.mean(rank_m[s] < 32):>12.3f} {np.mean(rank_m[s] == 0):>11.3f} {np.nanmedian(rank_f[s]):>21.0f} {np.mean(rank_f[s] < 32):>15.3f} {np.mean(rank_f[s] < 256):>16.3f} {np.nanmean(pct[ho]):>12.3f}")
print("\nBy word length (m=0 | m=2): pct(mask), top32(mask), unmasked top256")
for LL in range(2, 16):
    s0 = found & (m == 0) & (L == LL); s2 = found & (m == 2) & (L == LL)
    if s0.sum() == 0: continue
    print(f"L={LL:>2} n={s0.sum():>6} | m0: pct {np.nanmean(pct[s0]):.3f} top32 {np.mean(rank_m[s0] < 32):.3f} unm256 {np.mean(rank_f[s0] < 256):.3f} | m2: pct {np.nanmean(pct[s2]) if s2.any() else float('nan'):.3f} top32 {np.mean(rank_m[s2] < 32) if s2.any() else float('nan'):.3f} unm256 {np.mean(rank_f[s2] < 256) if s2.any() else float('nan'):.3f}")
print("\nWorst-retrieved words at m=0 (percentile inside mask, mask size):")
s0 = np.where(found & (m == 0))[0]; worst = s0[np.argsort(-pct[s0])[:40]]
print(", ".join(f"{lex.words[wi[i]]}({pct[i]:.2f},{int(msz[i])})" for i in worst))
never = sorted(set(wi[found & (m == 0) & (pct > 0.5)].tolist()))
print(f"\nwords ranked in the bottom half of their mask at m=0: {len(never)} of {lex.NW}")
