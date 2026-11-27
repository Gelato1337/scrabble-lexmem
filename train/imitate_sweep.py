"""Several policy nets trained at once on one stream of batches: the soft-target recipe of train/imitate.py (its EI=1, BETA=0 path:
same loss, optimiser, schedule and checks), each net with its own width, depth, learning rate and warm-up. The batches depend only
on the dataset seed, NWORK and B (lexmem/policy.py MoveDataset), and every net is initialised under seed 0, so each net trains
exactly as a single train/imitate.py run with its settings does. Net i sits on GPU i modulo the number of GPUs, one thread per net.
Env: SWEEP="tag:D:LAYERS:LR:WARMUP ..." (space-separated); TRAIN, TEST, LEXICON, STEPS, B, TAU, MARGIN, NWORK, NEV, EVAL_EVERY, KPLY,
EXCH as in train/imitate.py; NGPU (GPUs to spread the nets over, default 1); MAXH (stop at step 400 if the whole run would take
more than MAXH hours; 0 = never); STALL (no progress for this many seconds: print where every thread stands and stop, default 1200).
Writes data/policy_<tag>.pt at every check."""
import os, sys, time, math, pickle, random, faulthandler, numpy as np, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from lexmem.core import *; from lexmem.policy import *
import warnings; warnings.filterwarnings("ignore")
E = os.environ.get
STEPS = int(E("STEPS", 30000)); B = int(E("B", 48)); NW_ = int(E("NWORK", E("SLURM_CPUS_PER_TASK", 8))); TAU = float(E("TAU", 1.0)); MARGIN = float(E("MARGIN", 0.0))
NEV = int(E("NEV", 600)); EVAL_EVERY = int(E("EVAL_EVERY", 2000)); KPLY = int(E("KPLY", 4)); EXCH = E("EXCH", "0") == "1"; MAXH = float(E("MAXH", 0)); STALL = int(E("STALL", 1200))
assert E("SWEEP"), "SWEEP is not set"
faulthandler.dump_traceback_later(STALL, exit=True)   # a hung job bills until its time limit: stop it instead
ngpu = int(E("NGPU", 1)) if torch.cuda.is_available() else 0; dt = "cuda" if ngpu else "cpu"
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))
games_tr = [g for t in E('TRAIN', 'hasty_train').split(",") for g in pickle.load(open(f"data/games_{t}.pkl", "rb"))]   # TRAIN=a,b concatenates
games_te = pickle.load(open(f"data/games_{E('TEST','hasty_test')}.pkl", "rb"))
print(f"train games {len(games_tr)} test games {len(games_te)}; {ngpu} GPUs; {STEPS} steps of {B}", flush=True)

class Run: pass
runs = []
for i, c in enumerate(E("SWEEP").split()):
    r = Run(); r.tag, D, L, lr, wu = c.split(":"); r.dev = f"cuda:{i % ngpu}" if ngpu else "cpu"; r.la = None
    torch.manual_seed(0); random.seed(0); np.random.seed(0)   # the start of a single train/imitate.py run of this shape
    r.model = PolicyNet(lex, D=int(D), layers=int(L), use_lex=False).to(r.dev)
    r.opt = torch.optim.AdamW(r.model.parameters(), lr=float(lr), weight_decay=0.01)
    r.sched = torch.optim.lr_scheduler.LambdaLR(r.opt, lambda s, w=int(wu): min(1, (s + 1) / w) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    print(f"{r.tag}: D={D} layers={L} lr={lr} warm-up={wu} on {r.dev}, {sum(p.numel() for p in r.model.parameters() if p.requires_grad) / 1e6:.2f}M trainable parameters", flush=True)
    runs.append(r)
ds = MoveDataset(lex, games_tr, seed=0, use_lex=False, kply=KPLY, tau=TAU, exch=EXCH, ei=True)
dl = DataLoader(ds, batch_size=B, num_workers=NW_, collate_fn=collate_moves, persistent_workers=NW_ > 0, prefetch_factor=4 if NW_ > 0 else None)
ev = MoveDataset(lex, games_te, seed=1, use_lex=False, kply=KPLY, exch=EXCH, ei=True); ev.rng = random.Random(1); EV = [ev[i] for i in range(NEV)]

@torch.no_grad()
def evaluate(r, tag):
    model = r.model; model.eval(); acc = []; top3 = []; vmae = []; v0mae = []; vkbase = []
    for i in range(0, len(EV), 16):
        b = collate_moves(EV[i:i + 16]); b = {k: v.to(r.dev) for k, v in b.items()}
        with torch.autocast(dt, dtype=torch.bfloat16, enabled=dt == "cuda"): pl, v = model(b); _, v0 = model(b, with_moves=False)
        pl = pl.float(); order = pl.argsort(1, descending=True)
        acc += (order[:, 0] == b["target"]).tolist(); top3 += (order[:, :3] == b["target"][:, None]).any(1).tolist(); vmae += (v.float() - b["value"]).abs().tolist(); v0mae += (v0.float() - b["value_k"]).abs().tolist(); vkbase += b["value_k"].abs().tolist()
    model.train()
    return f"  [{r.tag} {tag}] hasty-move top1={np.mean(acc):.3f} top3={np.mean(top3):.3f} value MAE={np.mean(vmae)*100:.1f} pts | {KPLY}-ply spread MAE={np.mean(v0mae)*100:.1f} pts (predict-zero baseline {np.mean(vkbase)*100:.1f})"

def train_step(r, b):
    b = {k: v.to(r.dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dt, dtype=torch.bfloat16, enabled=dt == "cuda"): pl, v = r.model(b); _, v0 = r.model(b, with_moves=False)
    logp = torch.log_softmax(pl.float(), 1); wm = (b["margin"] >= MARGIN).float()   # soft target from the sim, as train/imitate.py with EI=1
    ce = -(b["soft"] * logp).sum(1); loss = (ce * wm).sum() / wm.sum().clamp_min(1)
    loss = loss + 0.5 * F.smooth_l1_loss(v.float(), b["value"]) + 1.0 * F.smooth_l1_loss(v0.float(), b["value_k"])
    r.opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(r.model.parameters(), 1.0); r.opt.step(); r.sched.step()
    x = loss.item(); r.la = x if r.la is None else 0.98 * r.la + 0.02 * x

it = iter(dl)   # the loader's workers start before the training threads exist
for r in runs: print(evaluate(r, "untrained"), flush=True)
pool = ThreadPoolExecutor(len(runs)); t0 = time.time(); step = 0
for b in it:
    if step < 3:   # the first steps one net after the other: each GPU sets up its kernels alone
        for r in runs: train_step(r, b)
    else: list(pool.map(lambda r: train_step(r, b), runs))
    step += 1
    if step == 100: t1 = time.time()
    if step % 100 == 0 or step < 3: faulthandler.dump_traceback_later(STALL, exit=True)
    if step % 100 == 0: print(f"step {step} t={time.time()-t0:.0f}s loss " + " ".join(f"{r.tag}={r.la:.3f}" for r in runs), flush=True)
    if MAXH and step == 400 and (time.time() - t1) / 300 * STEPS > MAXH * 3600:
        print(f"STOP: {(time.time() - t1) / 300:.2f} s per step, {(time.time() - t1) / 300 * STEPS / 3600:.1f} h for {STEPS} steps > MAXH={MAXH}", flush=True); os._exit(3)
    if step % EVAL_EVERY == 0 or step == STEPS:
        for line in list(pool.map(lambda r: evaluate(r, f"step {step}"), runs)): print(line, flush=True)
        for r in runs: torch.save(r.model.state_dict(), f"data/policy_{r.tag}.pt")
    if step >= STEPS: break
for line in list(pool.map(lambda r: evaluate(r, "final"), runs)): print(line, flush=True)
sys.stdout.flush(); os._exit(0)   # the nets are saved; no waiting on the loader's workers or the threads
