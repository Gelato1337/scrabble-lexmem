"""Sample positions for the planning benchmark and write them as CGP lines for go/lexmem_plan; merge its labels back.
  sample: python prep/plan_positions.py sample spf_0 200000 data/plan_in.jsonl   (positions with the mover to play, any phase)
  merge : python prep/plan_positions.py merge  spf_0 data/plan_in.jsonl data/plan_out.jsonl TAG -> data/plan_{TAG}.pkl
The pickle holds one dict per position: the position fields (board, blanks, rack, unseen, scores, bag, turn, mover, gi) and
cands = [(desc, score, E_opp, E_my, sd_opp, sd_my)] with desc = (0, r, c, d, L, widx, score, blanks)."""
import sys, json, random, pickle, types
from lexmem.harness import HastyBot
cgp_of = lambda r: HastyBot.cgp(types.SimpleNamespace(lexicon="NWL23"), r["board"], r["blanks"], r["rack"], r["scores"], 0)
mode, src = sys.argv[1], sys.argv[2]
games = pickle.load(open(f"data/games_{src}.pkl", "rb"))
if mode == "sample":
    n, out = int(sys.argv[3]), sys.argv[4]
    keys = [(gi, ti) for gi, (tr, _) in enumerate(games) for ti, r in enumerate(tr) if r["move"][0] == "play"]
    keys = random.Random(0).sample(keys, min(n, len(keys)))
    with open(out, "w") as f:
        for gi, ti in keys: f.write(json.dumps({"id": [gi, ti], "cgp": cgp_of(games[gi][0][ti])}) + "\n")
    print(f"wrote {len(keys)} positions to {out}")
else:
    labels, tag = sys.argv[4], sys.argv[5]; res = []; bad = 0
    for line in open(labels):
        o = json.loads(line)
        if o.get("err") or not o.get("cands"): bad += 1; continue
        gi, ti = o["id"]; r = dict(games[gi][0][ti]); r["gi"] = gi
        r["cands"] = [(tuple(c["d"]), c["d"][6], c["eo"], c["em"], c["so"], c["sm"]) for c in o["cands"]]
        res.append(r)
    pickle.dump(res, open(f"data/plan_{tag}.pkl", "wb")); print(f"{len(res)} labelled positions ({bad} failed) -> data/plan_{tag}.pkl")
