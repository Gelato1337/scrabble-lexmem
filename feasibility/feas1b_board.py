"""
Step 1b: alive words that also FIT the board.

Board: 15x15. We build plausible mid-game boards by simple greedy random self-play
(legal-ish placement, cross-words checked against lexicon).
Then, for a given (rack, U, board): a word is 'placeable' if some lane accepts it:
  lane = (row/col, start, direction, length) where the word overlays existing tiles
  consistently, touches at least one tile (or center on empty board), does not run
  into tiles beyond its ends, and the NEW letters it needs fit in rack (+U for future).
Cross-checks are ignored here -> this is an upper bound on placeability.
"""
import numpy as np, random, time
from collections import defaultdict

DIST = dict(A=9,B=2,C=2,D=4,E=12,F=2,G=3,H=2,I=9,J=1,K=1,L=4,M=2,N=6,O=8,
            P=2,Q=1,R=6,S=4,T=6,U=4,V=2,W=2,X=1,Y=2,Z=1)
LET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"; IDX={c:i for i,c in enumerate(LET)}
words=[w.strip().upper() for w in open("enable1.txt") if 2<=len(w.strip())<=15]
WSET=set(words)
W=np.zeros((len(words),26),dtype=np.int8)
for i,w in enumerate(words):
    for c in w: W[i,IDX[c]]+=1
by_len=defaultdict(list)
for i,w in enumerate(words): by_len[len(w)].append(i)
by_len={k:np.array(v) for k,v in by_len.items()}
WARR=np.array([list(w.ljust(15)) for w in words])   # (N,15) char array

N=15
def empty_board(): return [[None]*N for _ in range(N)]

def lanes(board):
    """yield (r,c,dr,dc,length,pattern) for every maximal-legal lane placement."""
    occupied=[(r,c) for r in range(N) for c in range(N) if board[r][c]]
    first = not occupied
    for dr,dc in ((0,1),(1,0)):
        for r in range(N):
            for c in range(N):
                # start must not be preceded by a tile
                pr,pc=r-dr,c-dc
                if 0<=pr<N and 0<=pc<N and board[pr][pc]: continue
                for L in range(2,16):
                    er,ec=r+dr*(L-1),c+dc*(L-1)
                    if not(0<=er<N and 0<=ec<N): break
                    # square after end must not be a tile
                    nr,nc=er+dr,ec+dc
                    if 0<=nr<N and 0<=nc<N and board[nr][nc]: continue
                    pat=[board[r+dr*i][c+dc*i] for i in range(L)]
                    n_fixed=sum(1 for p in pat if p)
                    if n_fixed==L: continue           # nothing to place
                    if L-n_fixed>7: continue          # can't place >7 tiles
                    touches = n_fixed>0
                    if not touches:
                        # must be adjacent to a tile, or cover center on empty board
                        if first:
                            touches = any((r+dr*i,c+dc*i)==(7,7) for i in range(L))
                        else:
                            for i in range(L):
                                rr,cc=r+dr*i,c+dc*i
                                for ar,ac in ((rr+dc,cc+dr),(rr-dc,cc-dr)):
                                    if 0<=ar<N and 0<=ac<N and board[ar][ac]: touches=True
                    if not touches: continue
                    yield r,c,dr,dc,L,pat

def placeable_mask(board, rack_counts, blanks_rack, avail_counts, blanks_avail, need_rack=0):
    """bool mask over words: fits some lane; new letters needed fit in avail (+blanks).
       need_rack: require >= this many new letters to come from rack (relevance)."""
    ok=np.zeros(len(words),dtype=bool)
    for r,c,dr,dc,L,pat in lanes(board):
        cand=by_len.get(L)
        if cand is None: continue
        sub=WARR[cand,:L]
        m=np.ones(len(cand),dtype=bool)
        for i,p in enumerate(pat):
            if p: m&=(sub[:,i]==p)
        if not m.any(): continue
        cand=cand[m]
        # letters to place = word letters minus fixed letters
        fixed=np.zeros(26,dtype=np.int16)
        for p in pat:
            if p: fixed[IDX[p]]+=1
        need=np.maximum(W[cand]-fixed,0)
        deficit=np.maximum(need-avail_counts,0).sum(1)
        good=deficit<=blanks_avail
        if need_rack:
            from_rack=np.minimum(need,rack_counts).sum(1)
            good&=from_rack>=need_rack
        ok[cand[good]]=True
    return ok

def counts(tiles):
    v=np.zeros(26,dtype=np.int16); b=0
    for t in tiles:
        if t=="?": b+=1
        else: v[IDX[t]]+=1
    return v,b

def cross_ok(board,r,c,dr,dc,word):
    """check perpendicular words formed by newly placed letters."""
    for i,ch in enumerate(word):
        rr,cc=r+dr*i,c+dc*i
        if board[rr][cc]: continue
        # perpendicular run
        pdr,pdc=dc,dr
        s=[]; ar,ac=rr-pdr,cc-pdc
        while 0<=ar<N and 0<=ac<N and board[ar][ac]: s.append(board[ar][ac]); ar-=pdr; ac-=pdc
        s=s[::-1]+[ch]; ar,ac=rr+pdr,cc+pdc
        while 0<=ar<N and 0<=ac<N and board[ar][ac]: s.append(board[ar][ac]); ar+=pdr; ac+=pdc
        if len(s)>1 and "".join(s) not in WSET: return False
    return True

def random_game(target_tiles_on_board, seed):
    """crude self-play: random legal placement using rack letters only, until N tiles on board."""
    rnd=random.Random(seed)
    bag=[c for c,n in DIST.items() for _ in range(n)]+["?","?"]; rnd.shuffle(bag)
    board=empty_board(); racks=[bag[:7],bag[7:14]]; bag=bag[14:]; on=0; turn=0; stuck=0
    while on<target_tiles_on_board and stuck<6:
        rack=racks[turn%2]; rc,rb=counts(rack)
        placed=False
        allL=list(lanes(board)); rnd.shuffle(allL)
        for r,c,dr,dc,L,pat in allL[:400]:
            cand=by_len.get(L)
            if cand is None: continue
            sub=WARR[cand,:L]; m=np.ones(len(cand),dtype=bool)
            for i,p in enumerate(pat):
                if p: m&=(sub[:,i]==p)
            cand=cand[m]
            if len(cand)==0: continue
            fixed=np.zeros(26,dtype=np.int16)
            for p in pat:
                if p: fixed[IDX[p]]+=1
            need=np.maximum(W[cand]-fixed,0)
            good=np.where(np.maximum(need-rc,0).sum(1)<=rb)[0]
            rnd.shuffle(list(good))
            for g in good[:20]:
                w=words[cand[g]]
                if not cross_ok(board,r,c,dr,dc,w): continue
                # place
                newrack=list(rack)
                for i,ch in enumerate(w):
                    rr,cc=r+dr*i,c+dc*i
                    if board[rr][cc]: continue
                    board[rr][cc]=ch; on+=1
                    if ch in newrack: newrack.remove(ch)
                    else: newrack.remove("?")
                draw=bag[:7-len(newrack)]; bag=bag[len(draw):]
                racks[turn%2]=newrack+draw; placed=True; break
            if placed: break
        if not placed:
            stuck+=1; rnd.shuffle(bag); racks[turn%2]=bag[:7]+[]; bag=bag[7:]+racks[turn%2][:0]
        else: stuck=0
        turn+=1
    unseen=bag+racks[(turn+1)%2]
    return board,racks[turn%2],unseen

def show(board):
    return "\n".join("".join(x or "." for x in row) for row in board)

if __name__=="__main__":
    print(f"{'tiles on board':>14} {'|U|':>4} {'lanes':>6} {'fits board (from rack+U)':>25} {'>=3 from rack':>14} {'>=5 from rack':>14} {'playable now':>13}  sec")
    for target in (0,8,20,35,50,65):
        rows=[]
        for seed in range(4):
            board,rack,unseen=random_game(target,seed)
            rc,rb=counts(rack); uc,ub=counts(unseen)
            t0=time.time()
            nl=sum(1 for _ in lanes(board))
            m_future=placeable_mask(board,rc,rb,rc+uc,rb+ub)
            m_k3=placeable_mask(board,rc,rb,rc+uc,rb+ub,need_rack=3)
            m_k5=placeable_mask(board,rc,rb,rc+uc,rb+ub,need_rack=5)
            m_now=placeable_mask(board,rc,rb,rc,rb)
            rows.append((len(unseen),nl,m_future.sum(),m_k3.sum(),m_k5.sum(),m_now.sum(),time.time()-t0))
        r=np.array(rows).mean(0)
        print(f"{target:>14} {r[0]:>4.0f} {r[1]:>6.0f} {r[2]:>25.0f} {r[3]:>14.0f} {r[4]:>14.0f} {r[5]:>13.0f}  {r[6]:.1f}")
    board,rack,unseen=random_game(35,1)
    print("\nexample board (35 tiles):"); print(show(board)); print("rack:",rack)
