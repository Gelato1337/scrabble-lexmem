"""Check the Go self-play engine (PARITY=1 output) against our Python pipeline, position by position:
1. move sets: Go's tokens (Macondo movegen, top 400 by score + exchanges) vs MoveDataset.build (our movegen): coverage and scores;
2. logits: the net on the Python-built item vs the logits the GPU server returned to Go, on the moves both contain.
Usage: CKPT=... python parity_check.py data/sp_parity.jsonl [max_positions]"""
import os, sys, json, numpy as np, torch
from lexmem.core import *; from lexmem.policy import PolicyNet, MoveDataset, collate_moves
torch.backends.mha.set_fastpath_enabled(False)
lex = Lexicon(os.environ.get("LEXICON", "data/NWL23.txt")); helper = MoveDataset(lex, [], exch=True)
model = PolicyNet(lex); model.load_state_dict(torch.load(os.environ["CKPT"], map_location="cpu"), strict=False); model.eval()
maxp = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cov_go, cov_py, score_ok, same_set, ldiff, corr, n = [], [], [], [], [], [], 0
for line in open(sys.argv[1]):
    g = json.loads(line)
    for p in g["pos"]:
        if "moves" not in p: continue
        board = np.full((N, N), -1, dtype=np.int64 if False else np.int16); blanks = np.zeros((N, N), dtype=bool)
        for i, ch in enumerate(p["b"]):
            if ch != ".": board[i // 15, i % 15] = IDX[ch.upper()]; blanks[i // 15, i % 15] = ch.islower()
        rec = dict(board=board, blanks=blanks, rack=np.array(p["rk"], dtype=np.int16), unseen=np.array(p["un"], dtype=np.int16), scores=tuple(p["sc"]), bag=p["bag"], turn=p["t"])
        it = helper.build(rec, 0.0)
        if it is None: continue
        py = {}
        for i, m in enumerate(it["_moves"]):
            py[("x",) + tuple(m[5]) if m[3] == 2 else (m[1], m[2], m[3], m[5])] = (i, m[0])
        go = {}
        for i, d in enumerate(p["moves"]):
            go[("x",) + tuple(d[1:]) if d[0] == 1 else (d[1], d[2], d[3], d[5])] = (i, d[6] if d[0] == 0 else 0)
        both = set(py) & set(go)
        cov_go.append(len(both) / len(go)); cov_py.append(len(both) / len(py)); same_set.append(set(py) == set(go))
        score_ok.append(np.mean([py[k][1] == go[k][1] for k in both]) if both else 1.0)
        with torch.no_grad(): pl, _ = model(collate_moves([it]))
        a = np.array([float(pl[0, py[k][0]]) for k in both]); b = np.array([p["logits"][go[k][0]] for k in both])
        if len(a) > 2: corr.append(np.corrcoef(a, b)[0, 1])
        if set(py) == set(go): ldiff.append(np.abs(a - b).max())
        n += 1
        if n >= maxp: break
    if n >= maxp: break
print(f"{n} positions: identical move sets {np.mean(same_set):.3f}; Go moves found by Python {np.mean(cov_go):.4f}; Python moves found by Go {np.mean(cov_py):.4f}; "
      f"scores agree {np.mean(score_ok):.4f}")
print(f"logits: corr (Python fp32 vs server bf16) median {np.median(corr):.4f} min {np.min(corr):.4f}; on identical move sets max |diff| median {np.median(ldiff) if ldiff else float('nan'):.3f}")
