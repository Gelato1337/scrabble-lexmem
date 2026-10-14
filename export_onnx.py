"""Spike for running our nets inside Go (Macondo): export to ONNX, check against PyTorch, report the speed budget.
1. value.onnx: position-only value pass -> (B, 2) [spread change / 100, win logit].
   Inputs: board (B,15,15) int64 (letter idx + 1, 0 = empty), rack (B,27), unseen (B,27), glob (B,3) [bag/100, spread/100, turn/30].
2. Parity: onnxruntime vs PyTorch on real positions (if onnxruntime is installed).
3. Budget: how many sim iterations / candidates the 10 s teacher ran per position (cand_iters in data/games_simsoft_31.pkl).
Usage: CKPT=data/policy_simsoft_exch_nolex.pt OUT=data/onnx python export_onnx.py"""
import os, time, pickle, numpy as np, torch, torch.nn as nn
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
torch.backends.mha.set_fastpath_enabled(False)
E = os.environ.get; OUT = E("OUT", "data/onnx"); os.makedirs(OUT, exist_ok=True)
lex = Lexicon(E("LEXICON", "data/NWL23.txt")); model = PolicyNet(lex); model.load_state_dict(torch.load(E("CKPT", "data/policy_simsoft_exch_nolex.pt"), map_location="cpu"), strict=False); model.eval()
class Value(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, board, rack, unseen, glob):
        _, v = self.m.qv(dict(board=board, rack=rack, unseen=unseen, glob=glob), with_moves=False); return v
games = pickle.load(open(f"data/games_{E('TEST', 'simsoft_31')}.pkl", "rb")); helper = MoveDataset(lex, [])
recs = [r for tr, _ in games[:200] for r in tr][:512]
b = collate_moves([helper.build_pos(r["board"], r["rack"], r["unseen"], r["scores"], r["bag"], r["turn"]) for r in recs])
args = (b["board"], b["rack"], b["unseen"], b["glob"]); vm = Value(model).eval()
with torch.no_grad(): ref = vm(*args).numpy()
torch.onnx.export(vm, args, f"{OUT}/value.onnx", input_names=["board", "rack", "unseen", "glob"], output_names=["value"],
                  dynamic_axes={k: {0: "batch"} for k in ["board", "rack", "unseen", "glob", "value"]}, opset_version=17)
print(f"exported {OUT}/value.onnx ({os.path.getsize(f'{OUT}/value.onnx')/1e6:.1f} MB)", flush=True)
try:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; sess = ort.InferenceSession(f"{OUT}/value.onnx", so, providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in zip(["board", "rack", "unseen", "glob"], args)}
    out = sess.run(None, feed)[0]; print(f"parity: max |onnx - torch| = {np.abs(out - ref).max():.2e}", flush=True)
    for bs in (1, 16, 64, 256):
        f = {k: v[:bs] for k, v in feed.items()}; sess.run(None, f); t = time.time(); n = 0
        while time.time() - t < 2: sess.run(None, f); n += 1
        dt = (time.time() - t) / n; print(f"onnxruntime python, 1 thread, batch {bs:4d}: {dt*1e3:7.2f} ms/batch  {dt/bs*1e6:8.1f} us/position", flush=True)
except ImportError: print("onnxruntime not installed in this environment (parity/timing skipped)", flush=True)
torch.set_num_threads(1)
for bs in (1, 16, 64):
    a = tuple(x[:bs] for x in args); vm(*a); t = time.time(); n = 0
    with torch.no_grad():
        while time.time() - t < 2: vm(*a); n += 1
    dt = (time.time() - t) / n; print(f"pytorch, 1 thread, batch {bs:4d}: {dt*1e3:7.2f} ms/batch  {dt/bs*1e6:8.1f} us/position", flush=True)
it = [(r["cand_iters"], len(r["cands"])) for tr, _ in games for r in tr if r.get("known") and len(r.get("cands", [])) > 1]
iters = np.array([max(x[0]) for x in it]); nc = np.array([x[1] for x in it])
print(f"teacher (SIMMING_BOT 10 s, 1 thread): median {np.median(iters):.0f} iterations of the top candidate, median {np.median(nc):.0f} candidates reported "
      f"-> ~{np.median(iters)*np.median(nc)/10:.0f} candidate-rollouts/s; one horizon value per rollout needs < {1e6/(np.median(iters)*np.median(nc)/10):.0f} us/position to cost < 100 % extra", flush=True)
# reference inputs/outputs for parity checks in Go (64 real positions; raw little-endian arrays)
n = 64
for name, arr in zip(["board", "rack", "unseen", "glob"], args): arr[:n].numpy().astype(np.int64 if name == "board" else np.float32).tofile(f"{OUT}/ref_{name}.bin")
ref[:n].astype(np.float32).tofile(f"{OUT}/ref_value.bin"); print(f"wrote {OUT}/ref_*.bin for {n} positions", flush=True)
