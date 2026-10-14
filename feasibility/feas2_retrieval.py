"""
Step 2 (CPU, scaled down): lexicon memory layer as a soft-GADDAG.

Keys   : per-word vector = letter-composition encoder(word) + learned per-word slot
Query  : MLP(rack counts, blanks, unseen counts, lane pattern)
Score  : dot(query, key), masked by the rules (pattern fits, letters available)
Train  : constructed positions. Two modes:
   NOW     : all letters the word needs are on the rack. Label = ALL words that fit
             the lane from the rack (multi-positive). "Soft-GADDAG" test.
   DELAYED : 1-2 needed letters are moved from rack to U. Label = the seed word.
             "Does the query reach for a word it cannot play yet?" test.
Eval   : recall@k over the full masked candidate set (the set the real system would rank).
"""
import numpy as np, random, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.manual_seed(0); random.seed(0); np.random.seed(0)

DIST = dict(A=9,B=2,C=2,D=4,E=12,F=2,G=3,H=2,I=9,J=1,K=1,L=4,M=2,N=6,O=8,
            P=2,Q=1,R=6,S=4,T=6,U=4,V=2,W=2,X=1,Y=2,Z=1)
LET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"; IDX={c:i for i,c in enumerate(LET)}
MAXL=8
words=[w.strip().upper() for w in open("enable1.txt") if 2<=len(w.strip())<=MAXL]
NW=len(words); print(f"lexicon subset: {NW} words (len<=8)")
W=np.zeros((NW,26),dtype=np.int16)
POS=np.zeros((NW,MAXL),dtype=np.int64)     # letter index per position, 26 = pad
POS[:]=26
LEN=np.zeros(NW,dtype=np.int64)
for i,w in enumerate(words):
    LEN[i]=len(w)
    for j,c in enumerate(w): W[i,IDX[c]]+=1; POS[i,j]=IDX[c]
by_len={L:np.where(LEN==L)[0] for L in range(2,MAXL+1)}
Wt=torch.tensor(W,dtype=torch.float32); POSt=torch.tensor(POS); LENt=torch.tensor(LEN)
BAG=[c for c,n in DIST.items() for _ in range(n)]

D=96
class WordEnc(nn.Module):            # composition encoder + per-word slot
    def __init__(s):
        super().__init__()
        s.pos=nn.Embedding(27,32); s.posw=nn.Parameter(torch.randn(MAXL,32)*0.1)
        s.mlp=nn.Sequential(nn.Linear(26+32+1,256),nn.GELU(),nn.Linear(256,D))
        s.slot=nn.Embedding(NW,D); nn.init.normal_(s.slot.weight,std=0.02)
    def forward(s,idx):
        cnt=Wt[idx]/3.0
        p=(s.pos(POSt[idx])*s.posw).sum(1)
        x=torch.cat([cnt,p,LENt[idx,None].float()/8],1)
        return s.mlp(x)+s.slot(idx)
class QueryEnc(nn.Module):
    def __init__(s):
        super().__init__()
        s.pat=nn.Embedding(28,32)            # 26 letters, 27=empty-in-lane, 28-1=off-lane handled by zero
        s.patw=nn.Parameter(torch.randn(MAXL,32)*0.1)
        s.mlp=nn.Sequential(nn.Linear(26+1+26+32+1,384),nn.GELU(),nn.Linear(384,384),nn.GELU(),nn.Linear(384,D))
    def forward(s,rack,blanks,unseen,pat,L):
        p=(s.pat(pat)*s.patw).sum(1)
        x=torch.cat([rack/3.0,blanks[:,None]/2,unseen/12.0,p,L[:,None].float()/8],1)
        return s.mlp(x)

def construct(mode):
    """returns rack(26), blanks, unseen(26), pat(MAXL), L, label_idx(np array), seed"""
    seed=random.randrange(NW); w=words[seed]; L=len(w)
    nfix=random.randint(0,min(2,L-1))
    fixed=set(random.sample(range(L),nfix))
    pat=np.full(MAXL,27,dtype=np.int64)
    for j in range(L):
        pat[j]=IDX[w[j]] if j in fixed else 27
    pat[L:]=27  # (positions beyond L are irrelevant; L is given separately)
    need=[w[j] for j in range(L) if j not in fixed]
    bag=BAG[:]; random.shuffle(bag)
    rack=list(need); blanks=0
    if mode=="DELAYED":
        k=random.randint(1,min(2,len(rack)))
        moved=[rack.pop(random.randrange(len(rack))) for _ in range(k)]
    else: moved=[]
    if random.random()<0.25 and rack:      # sometimes a blank replaces a needed letter
        rack.pop(random.randrange(len(rack))); blanks+=1
    while len(rack)+blanks<7: rack.append(bag.pop())
    # unseen: rest of bag minus what's on rack (rough), plus moved letters guaranteed in
    for t in rack:
        if t in bag: bag.remove(t)
    unseen=bag[:60]+moved
    rc=np.zeros(26,dtype=np.float32); uc=np.zeros(26,dtype=np.float32)
    for t in rack: rc[IDX[t]]+=1
    for t in unseen: uc[IDX[t]]+=1
    # rules mask: pattern fits & needed letters available (from rack for NOW; rack+unseen for DELAYED)
    cand=by_len[L]
    m=np.ones(len(cand),dtype=bool)
    for j in range(L):
        if pat[j]<26: m&=(POS[cand,j]==pat[j])
    cand=cand[m]
    fixedc=np.zeros(26,dtype=np.int16)
    for j in fixed: fixedc[IDX[w[j]]]+=1
    needm=np.maximum(W[cand]-fixedc,0)
    if mode=="NOW":
        ok=np.maximum(needm-rc,0).sum(1)<=blanks
        mask=cand[ok]; label=mask          # every word playable from the rack is a positive
    else:
        avail=rc+uc
        ok=(np.maximum(needm-avail,0).sum(1)<=blanks)&(np.minimum(needm,rc).sum(1)>=max(1,L-nfix-2))
        mask=cand[ok]; label=np.array([seed])
    return rc,blanks,uc,pat,L,mask,label,seed

wenc=WordEnc(); qenc=QueryEnc()
opt=torch.optim.AdamW(list(wenc.parameters())+list(qenc.parameters()),lr=2e-3,weight_decay=1e-4)
NEG=1024; B=32

def batch_loss(mode):
    tot=0.0
    for _ in range(B):
        rc,bl,uc,pat,L,mask,label,seed=construct(mode)
        if len(mask)==0: continue
        labs=label[:64] if len(label)>64 else label
        negs=np.random.choice(mask,size=min(NEG,len(mask)),replace=False)
        idx=np.unique(np.concatenate([labs,negs]))
        isl=torch.tensor(np.isin(idx,labs),dtype=torch.bool)
        q=qenc(torch.tensor(rc)[None],torch.tensor([bl],dtype=torch.float32),torch.tensor(uc)[None],torch.tensor(pat)[None],torch.tensor([L]))
        k=wenc(torch.tensor(idx))
        logits=(k@q[0])/np.sqrt(D)
        lp=F.log_softmax(logits,0)
        tot=tot-torch.logsumexp(lp[isl],0)      # multi-positive: -log P(any label)
    return tot/B

@torch.no_grad()
def evaluate(mode,n=60,ks=(8,32,64)):
    rec={k:[] for k in ks}; seedrank=[]; msz=[]
    for _ in range(n):
        rc,bl,uc,pat,L,mask,label,seed=construct(mode)
        if len(mask)==0: continue
        q=qenc(torch.tensor(rc)[None],torch.tensor([bl],dtype=torch.float32),torch.tensor(uc)[None],torch.tensor(pat)[None],torch.tensor([L]))
        k=wenc(torch.tensor(mask)); s=(k@q[0]).numpy()
        order=mask[np.argsort(-s)]
        for kk in ks:
            top=set(order[:kk].tolist()); rec[kk].append(len(top&set(label.tolist()))/min(len(label),kk))
        seedrank.append(int(np.where(order==seed)[0][0])+1 if seed in mask else -1)
        msz.append(len(mask))
    return {k:np.mean(v) for k,v in rec.items()}, np.median([r for r in seedrank if r>0]), np.mean(msz)

print("untrained:", evaluate("NOW"), evaluate("DELAYED"))
t0=time.time(); step=0; BUDGET=int(__import__("os").environ.get("BUDGET","220"))
while time.time()-t0<BUDGET:
    mode="NOW" if random.random()<0.6 else "DELAYED"
    loss=batch_loss(mode); opt.zero_grad(); loss.backward(); opt.step(); step+=1
    if step%40==0:
        rN,srN,mN=evaluate("NOW",40); rD,srD,mD=evaluate("DELAYED",40)
        print(f"step {step:4d} t={time.time()-t0:4.0f}s loss={loss.item():.3f} | NOW recall@8/32/64={rN[8]:.2f}/{rN[32]:.2f}/{rN[64]:.2f} mask~{mN:.0f} seed-rank~{srN:.0f} | DELAYED seed-in-top8/32/64={rD[8]:.2f}/{rD[32]:.2f}/{rD[64]:.2f} mask~{mD:.0f} seed-rank~{srD:.0f}",flush=True)
print("final NOW:",evaluate("NOW",100)); print("final DELAYED:",evaluate("DELAYED",100))
# coverage: for every word, does a NOW query built around it put it in the top-32?
hits=0; N=1500
with torch.no_grad():
    for i in np.random.choice(NW,N,replace=False):
        w=words[i]; L=len(w)
        rc=np.zeros(26,dtype=np.float32)
        for c in w: rc[IDX[c]]+=1
        filler=random.sample(BAG,7-min(7,L))
        for c in filler: rc[IDX[c]]+=1
        pat=np.full(MAXL,27,dtype=np.int64)
        cand=by_len[L]; ok=np.maximum(W[cand]-rc,0).sum(1)<=0; mask=cand[ok]
        q=qenc(torch.tensor(rc)[None],torch.zeros(1),torch.zeros(1,26),torch.tensor(pat)[None],torch.tensor([L]))
        s=(wenc(torch.tensor(mask))@q[0]).numpy(); order=mask[np.argsort(-s)]
        hits+= i in set(order[:32].tolist())
print(f"coverage: {hits/N:.3f} of random words retrieved in top-32 when they are the target (no board letters)")
torch.save({"wenc":wenc.state_dict(),"qenc":qenc.state_dict()},"lexmem_small.pt")
