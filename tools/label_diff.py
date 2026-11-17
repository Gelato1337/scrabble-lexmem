"""Compare two labellings of the same planning positions (e.g. data/plan_v1.pkl from the old lexmem_plan vs data/plan_h2.pkl from the
fixed one), by phase; with two-turn labels also how often the two-turn planning answer differs from the one-turn one.
Usage: python tools/label_diff.py data/plan_v1.pkl data/plan_h2.pkl"""
import sys, pickle, numpy as np
old, new = (pickle.load(open(f, "rb")) for f in sys.argv[1:3])
ref = {(r["gi"], r["turn"]): r for r in old}
PH = ((61, 100, "opening"), (31, 60, "middle"), (8, 30, "late"), (0, 7, "pre-end/end"))
rows = []   # per candidate: phase bag, d_eo, d_em ; per position: one-turn answer changed
pos = []
for r in new:
    q = ref.get((r["gi"], r["turn"]))
    if q is None: continue
    oc = {c[0]: c for c in q["cands"]}; js = [c for c in r["cands"] if c[0] in oc]
    if len(js) < 2: continue
    for c in js: rows.append((r["bag"], c[2] - oc[c[0]][2], c[3] - oc[c[0]][3]))
    sc = np.array([c[1] for c in js], float); pv_new = sc - [c[2] for c in js] + [c[3] for c in js]
    pv_old = sc - [oc[c[0]][2] for c in js] + [oc[c[0]][3] for c in js]
    two = len(js[0]) > 6
    pv2 = sc - [c[2] for c in js] + [c[6] for c in js] - [c[7] for c in js] if two else pv_new
    b1, b2, base = int(np.argmax(pv_new)), int(np.argmax(pv2)), int(np.argmax(sc))
    pos.append((r["bag"], int(np.argmax(pv_old)) != b1, b2 != b1, pv2[b2] - pv2[b1], pv2[b2] - pv2[base] >= 5, pv_new[b1] - pv_new[base] >= 5))
R, P = np.array(rows, float), np.array(pos, float)
print(f"{len(P)} positions, {len(R)} candidates matched")
for lo, hi, nm in ((0, 100, "all"),) + PH:
    k = (R[:, 0] >= lo) & (R[:, 0] <= hi); kp = (P[:, 0] >= lo) & (P[:, 0] <= hi)
    print(f"{nm:>12}: n_pos {int(kp.sum()):6d} | new - old E_opp mean {R[k,1].mean():+.2f} (|.| {np.abs(R[k,1]).mean():.2f}), E_my mean {R[k,2].mean():+.2f} (|.| {np.abs(R[k,2]).mean():.2f})"
          f" | one-turn answer changed {P[kp,1].mean():.3f} | two-turn answer != one-turn {P[kp,2].mean():.3f} (one-turn answer loses {P[kp,3].mean():.2f} PV2 pts)"
          f" | critical: one-turn {P[kp,5].mean():.3f}, two-turn {P[kp,4].mean():.3f}")
