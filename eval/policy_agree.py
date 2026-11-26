"""Policy nets against the teacher on the same held-out positions (one forward pass per position, no search).
Positions: N of data/games_<TEST>.pkl (teacher labels from prep/label_bot.py: SIMMING_BOT 10 s, its sim candidates with win %),
a fixed random sample (seed 7) of the turns where the teacher played tiles and listed >= 2 candidates. Moves offered to the net as
in training (train/imitate.py: our move generator's top MAXM plays by score, the teacher's move always kept, order shuffled).
Per checkpoint: agreement = the net's top move is the teacher's move; top 8 = the teacher's move is among the net's 8 highest;
loss = the teacher's best win % minus the teacher's win % for the net's top move, in win-% points (a move outside the teacher's
candidates counts at its lowest listed win %: a lower bound). Every checkpoint after the first is also compared with the first,
paired over positions (mean +- standard error). Width and depth are read from each checkpoint.
Usage: TEST=simsoft_31 N=5000 python eval/policy_agree.py REF.pt NEW.pt [...]      (env: EXCH, NWORK, LEXICON, OUT=file.pkl)"""
import os, sys, pickle, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from lexmem.core import *; from lexmem.policy import *
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
TEST = E("TEST", "simsoft_31"); NPOS = int(E("N", 5000)); NW = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); EXCH = E("EXCH", "0") == "1"
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); games = pickle.load(open(f"data/games_{TEST}.pkl", "rb"))
keys = [(gi, ti) for gi, (traj, sc) in enumerate(games) for ti, r in enumerate(traj)
        if r["move"][0] == "play" and r.get("known", True) and len(r.get("cands", ())) >= 2]
keys = random.Random(7).sample(keys, min(NPOS, len(keys)))
helper = MoveDataset(lex, games, use_lex=False, exch=EXCH)
class DS(Dataset):
    def __len__(self): return len(keys)
    def __getitem__(self, k):
        gi, ti = keys[k]; traj, sc = games[gi]; rec = traj[ti]; m = rec["move"]
        helper.rng = random.Random(gi * 1000 + ti)   # the shuffle of the move order: fixed per position
        it = helper.build(rec, 0.0, target_rcd=(m[1], m[2], m[3], m[5]))
        if it is None: return None
        idx = {}
        for j, mv in enumerate(it["_moves"]): idx.setdefault((mv[1], mv[2], mv[3], mv[5]), j)
        est = np.full(len(it["_moves"]), np.nan, dtype=np.float32)
        for (r_, c_, d_, L_, w_), e in zip(rec["cands"], rec["cand_est"]):
            j = idx.get((r_, c_, d_, w_))
            if j is not None: est[j] = e if np.isnan(est[j]) else max(est[j], e)
        if np.isnan(est).all(): return None
        it["est"] = est; it["bag"] = rec["bag"]; return it
items = [x for ch in DataLoader(DS(), batch_size=64, num_workers=NW, collate_fn=lambda b: b) for x in ch if x is not None]
print(f"{len(items)} of {len(keys)} sampled positions of games_{TEST} built (teacher played tiles, >= 2 candidates)", flush=True)

@torch.no_grad()
def run(path):
    sd = torch.load(path, map_location=dev); shape = net_shape(sd); model = PolicyNet(lex, use_lex=False, **shape).to(dev)
    res = model.load_state_dict(sd, strict=False); bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if not k.startswith(("qh.", "vh."))]
    assert not bad, f"{path}: checkpoint does not fit the net: {bad[:5]}"
    model.eval(); rows = []
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]; b = {k: v.to(dev) for k, v in collate_moves(chunk).items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): pl, _ = model(b)
        pl = pl.float().cpu()
        for j, it in enumerate(chunk):
            n = len(it["_moves"]); order = pl[j, :n].argsort(descending=True).numpy(); est = it["est"]; c = int(order[0])
            got = est[c] if not np.isnan(est[c]) else np.nanmin(est)
            rows.append((c == it["target"], it["target"] in order[:8], float(np.nanmax(est) - got), it["bag"]))
    return shape, sum(p.numel() for p in model.parameters()) / 1e6, np.array(rows, dtype=float)
def se(x): return x.std(ddof=1) / np.sqrt(len(x))
out = {}; base = None
for path in sys.argv[1:]:
    shape, npar, R = run(path); out[path] = R
    line = (f"{os.path.basename(path)} (width {shape['D']}, {shape['layers']} layers, {npar:.1f}M parameters): agreement {100 * R[:, 0].mean():.1f} % +- {100 * se(R[:, 0]):.1f}, "
            f"top 8 {100 * R[:, 1].mean():.1f} %, loss {R[:, 2].mean():.3f} +- {se(R[:, 2]):.3f}")
    for lo, hi, nm in ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (0, 7, "bag <= 7")):
        k = (R[:, 3] >= lo) & (R[:, 3] <= hi)
        if k.sum(): line += f" | {nm} (n={int(k.sum())}): {100 * R[k, 0].mean():.1f} %, loss {R[k, 2].mean():.3f}"
    if base is not None:
        d0, d2 = R[:, 0] - base[:, 0], R[:, 2] - base[:, 2]
        line += f"\n    vs {os.path.basename(sys.argv[1])}: agreement {100 * d0.mean():+.2f} +- {100 * se(d0):.2f} points, loss {d2.mean():+.3f} +- {se(d2):.3f} (negative = closer to the teacher)"
    else: base = R
    print(line, flush=True)
if E("OUT"): pickle.dump(out, open(E("OUT"), "wb"))
