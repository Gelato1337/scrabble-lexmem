"""
Step 1: how sparse is the reachable-word set under the exact rules mask?

Definitions (no board yet):
  U      = unseen multiset (bag + opp rack), exact
  R      = my rack (7 tiles)
  alive_U(w)   : letters(w) fits in R + U            (could ever be built)
  alive_k(w)   : alive_U and w uses >= k letters from R   (relevant to *me*)
  alive_now(w) : letters(w) fits in R alone         (playable from rack, no board help)

We simulate game stages by drawing tiles from the standard bag.
Blanks are treated as wildcards (2 max).
"""
import numpy as np, random, time
from collections import Counter

DIST = dict(A=9,B=2,C=2,D=4,E=12,F=2,G=3,H=2,I=9,J=1,K=1,L=4,M=2,N=6,O=8,
            P=2,Q=1,R=6,S=4,T=6,U=4,V=2,W=2,X=1,Y=2,Z=1)
LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX = {c:i for i,c in enumerate(LET)}

words = [w.strip().upper() for w in open("enable1.txt") if 2 <= len(w.strip()) <= 15]
W = np.zeros((len(words), 26), dtype=np.int8)          # letter count matrix
for i,w in enumerate(words):
    for c in w: W[i, IDX[c]] += 1
L = W.sum(1)
print(f"lexicon: {len(words)} words, length 2..15")

def counts(tiles):
    v = np.zeros(26, dtype=np.int16)
    for t in tiles:
        if t != "?": v[IDX[t]] += 1
    return v, tiles.count("?")

def alive(avail_counts, blanks, need_from_rack=None, rack_counts=None):
    """words whose letter needs fit in avail (+blanks as wildcards)."""
    deficit = np.maximum(W - avail_counts, 0).sum(1)       # letters missing
    ok = deficit <= blanks
    if need_from_rack is not None:
        from_rack = np.minimum(W, rack_counts).sum(1)
        ok &= from_rack >= need_from_rack
    return ok

full_bag = [c for c,n in DIST.items() for _ in range(n)] + ["?","?"]
random.seed(0)
stages = [100, 80, 60, 40, 20, 10, 0]   # tiles left in bag (before my rack is drawn)
print(f"\n{'bag':>4} {'|U|':>4} {'alive_U':>8} {'>=1 mine':>9} {'>=3 mine':>9} {'>=5 mine':>9} {'rack-only':>10}  ms")
for bag_left in stages:
    rows=[]
    for trial in range(20):
        bag = full_bag[:]; random.shuffle(bag)
        # tiles already on board = 100 - bag_left - 14 (two racks)
        n_board = max(0, 100 - bag_left - 14)
        board_tiles = bag[:n_board]; rest = bag[n_board:]
        rack = rest[:7]; unseen = rest[7:]            # unseen = opp rack + bag
        t0=time.time()
        rc, rb = counts(rack); uc, ub = counts(unseen)
        a_U  = alive(rc+uc, rb+ub)
        a_1  = alive(rc+uc, rb+ub, 1, rc)
        a_3  = alive(rc+uc, rb+ub, 3, rc)
        a_5  = alive(rc+uc, rb+ub, 5, rc)
        a_now= alive(rc, rb)
        ms=(time.time()-t0)*1000
        rows.append((len(unseen), a_U.sum(), a_1.sum(), a_3.sum(), a_5.sum(), a_now.sum(), ms))
    r = np.array(rows).mean(0)
    print(f"{bag_left:>4} {r[0]:>4.0f} {r[1]:>8.0f} {r[2]:>9.0f} {r[3]:>9.0f} {r[4]:>9.0f} {r[5]:>10.0f}  {r[6]:.1f}")

# How much does one draw change the alive set? (smoothness across turns)
print("\nturn-to-turn stability of alive_{>=3 mine}: Jaccard(before, after playing 4 tiles + drawing 4)")
jacs=[]
for trial in range(30):
    bag = full_bag[:]; random.shuffle(bag)
    rest = bag[30:]; rack = rest[:7]; unseen = rest[7:]
    rc,rb = counts(rack); uc,ub = counts(unseen)
    before = alive(rc+uc, rb+ub, 3, rc)
    keep = rack[4:]; draw = unseen[:4]; unseen2 = unseen[4:]
    rack2 = keep+draw
    rc2,rb2 = counts(rack2); uc2,ub2 = counts(unseen2)
    after = alive(rc2+uc2, rb2+ub2, 3, rc2)
    jacs.append((before&after).sum()/max(1,(before|after).sum()))
print(f"mean Jaccard = {np.mean(jacs):.3f}")
