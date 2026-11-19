"""Check the GPU retrieval path (PolicyNet.rok, used by selfplay/server_infer.py) against the training path (retrieval_mask
index lists, collate_moves): on N positions of data/games_<TEST>.pkl, (1) the dense masks select exactly the index lists' words,
(2) a lexicon net's policy logits and value agree between the two paths (bf16 autocast as in the server).
Usage: CKPT=data/policy_simsoft_scratch_lex.pt TEST=simsoft_31 N=512 PYTHONPATH=. python eval/check_rok.py"""
import os, pickle, random, numpy as np, torch
from lexmem.core import Lexicon
from lexmem.policy import PolicyNet, MoveDataset, collate_moves, RET_LENS
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
lex = Lexicon("data/NWL23.txt"); games = pickle.load(open(f"data/games_{E('TEST', 'simsoft_31')}.pkl", "rb"))
model = PolicyNet(lex, use_lex=True).to(dev); model.load_state_dict(torch.load(E("CKPT"), map_location=dev), strict=False); model.eval()
ds = MoveDataset(lex, games, seed=3, use_lex=True); ds.rng = random.Random(3); items = [ds[i] for i in range(int(E("N", 512)))]
nmask = 0; bad = 0; dl = []; dv = []; agree = []
with torch.no_grad():
    for i in range(0, len(items), 64):
        b = {k: v.to(dev) for k, v in collate_moves(items[i:i + 64]).items()}
        ok = model.rok(b["rack"], b["unseen"])
        for li, L in enumerate(RET_LENS):   # (1) the same word sets
            cand = getattr(model, f"cand{L}").cpu().numpy(); okn = ok[f"rok{li}"].cpu().numpy()
            for j, it in enumerate(items[i:i + 64]):
                nmask += 1; bad += set(cand[okn[j]].tolist()) != set(np.asarray(it["rmask"][li]).tolist())
        bd = {k: v for k, v in b.items() if not k.startswith("rmask") and not k.startswith("rvalid")}; bd.update(ok)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            p1, v1 = model(b); p2, v2 = model(bd)
        m = b["valid"]; dl.append((p1.float() - p2.float()).abs()[m].max().item()); dv.append((v1.float() - v2.float()).abs().max().item())
        agree += (p1.float().argmax(1) == p2.float().argmax(1)).tolist()
print(f"masks: {nmask - bad}/{nmask} identical word sets")
print(f"policy logits max |diff| {max(dl):.4f}, value max |diff| {max(dv):.4f}, top move agrees {np.mean(agree):.4f} on {len(agree)} positions")
