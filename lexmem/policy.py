"""Policy/value net over legal moves, with an optional lexicon-retrieval branch (the ablation switch)."""
import numpy as np, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from lexmem.core import *
from lexmem.data import LANE_F, MAXL
from lexmem.model import WordEncoder, QueryEncoder

MF = 8 + 27          # scalar move features + leave counts
MAXM = 400           # moves kept per position (top by score, target always kept)
KRET = 32            # retrieved word tokens per query
RET_LENS = (7, 8)    # rack-level queries: "which 7s / 8s can I reach"

def move_feats(lex, rack, moves):
    """moves: list of (score, r, c, d, L, widx). returns (M, MF) float32 and leave (M,27)."""
    M = len(moves); x = np.zeros((M, MF), dtype=np.float32)
    for i, (s, r, c, d, L, w) in enumerate(moves):
        x[i, 0] = s / 50; x[i, 1] = L / 15; x[i, 2] = r / 14; x[i, 3] = c / 14; x[i, 4] = d
        x[i, 5] = (s >= 50)  # bingo-ish
    return x

def retrieval_mask(lex, rack, unseen, L, maxmiss=3):
    """words of length L whose letters fit rack+unseen with <= maxmiss letters not on the rack (blanks cover)."""
    cand = lex.by_len[L]; W = lex.W_L[L]
    rb = int(rack[26]); deficit = np.maximum(W - rack[:26][None], 0); dsum = deficit.sum(1)
    avail = np.maximum(deficit - unseen[:26][None], 0).sum(1) <= rb
    ok = avail & (dsum - rb <= maxmiss)
    return cand[ok]

class MoveDataset(Dataset):
    """position -> legal moves (our movegen), target = index of the move actually played (HastyBot), value = final spread."""
    def __init__(self, lex, games, n_items=10**7, seed=0, use_lex=False, kply=4, tau=2.0):
        self.lex, self.games, self.n, self.seed, self.rng, self.use_lex, self.kply, self.tau = lex, games, n_items, seed, None, use_lex, kply, tau
        self.items = [(gi, ti) for gi, (traj, sc) in enumerate(games) for ti in range(len(traj)) if traj[ti]["move"][0] == "play"]
    def __len__(self): return self.n
    def build(self, rec, final, target_letters=None, target_rcd=None, moves=None):
        lex = self.lex; board, blanks, rack, unseen = rec["board"], rec["blanks"], rec["rack"], rec["unseen"]
        if moves is None:
            cc = cross_checks(lex, board, blanks); moves = legal_moves(lex, board, blanks, cc, rack)
        else: moves = [m[:6] for m in moves]
        if not moves: return None
        tgt = -1
        if target_rcd is not None:
            r0, c0, d0, letters = target_rcd
            word = "".join((LET[board[r0 + ((0,1),(1,0))[d0][0]*j, c0 + ((0,1),(1,0))[d0][1]*j]] if ch == "." else ch.upper()) for j, ch in enumerate(letters))
            for i, (s, r, c, d, L, w) in enumerate(moves):
                if (r, c, d) == (r0, c0, d0) and lex.words[w] == word: tgt = i; break
            if tgt < 0: return None
        moves.sort(key=lambda m: -m[0])
        if target_rcd is not None:
            tm = [m for m in moves if (m[1], m[2], m[3]) == (r0, c0, d0) and lex.words[m[5]] == word][0]
            moves = [tm] + [m for m in moves[:MAXM] if m is not tm]
            perm = list(range(len(moves))); (self.rng or random).shuffle(perm); moves = [moves[p] for p in perm]; tgt = perm.index(0)
        else: moves = moves[:MAXM]
        mf = move_feats(lex, rack, moves)
        used = np.zeros((len(moves), 27), dtype=np.float32)
        for i, (s, r, c, d, L, w) in enumerate(moves):
            fixed = np.zeros(26, dtype=np.int16); dr, dc = ((0, 1), (1, 0))[d]
            for j in range(L):
                b = board[r + dr * j, c + dc * j]
                if b >= 0: fixed[b] += 1
            need = np.maximum(lex.W[w] - fixed, 0); fr = np.minimum(need, rack[:26]); nb = int((need - fr).sum())
            used[i, :26] = fr; used[i, 26] = nb; mf[i, 6] = need.sum() / 7; mf[i, 7] = nb
        leave = (rack[None].astype(np.float32) - used)
        out = dict(_moves=moves, board=board.astype(np.int64) + 1, rack=rack.astype(np.float32), unseen=unseen.astype(np.float32),
                   glob=np.array([rec["bag"] / 100, (rec["scores"][0] - rec["scores"][1]) / 100, rec["turn"] / 30], dtype=np.float32),
                   mf=mf, leave=leave, widx=np.array([m[5] for m in moves], dtype=np.int64), target=tgt, value=np.float32(final / 100))
        if self.use_lex:
            out["rmask"] = [retrieval_mask(lex, rack, unseen, L) for L in RET_LENS]
        return out
    def build_pos(self, board, rack, unseen, scores, bag, turn):
        """position features only (no legal moves) - for value-only evaluation."""
        out = dict(board=board.astype(np.int64) + 1, rack=rack.astype(np.float32), unseen=unseen.astype(np.float32),
                   glob=np.array([bag / 100, (scores[0] - scores[1]) / 100, turn / 30], dtype=np.float32))
        if self.use_lex: out["rmask"] = [retrieval_mask(self.lex, rack, unseen, L) for L in RET_LENS]
        return out
    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + i)
        while True:
            gi, ti = self.items[self.rng.randrange(len(self.items))]; traj, sc = self.games[gi]; rec = traj[ti]
            m = rec["move"]; final = sc[rec["mover"]] - sc[1 - rec["mover"]]
            it = self.build(rec, final, target_rcd=(m[1], m[2], m[3], m[5]))
            if it is not None:
                it["w_pol"] = np.float32(0.0 if rec.get("explore") else 1.0)
                if "cands" in rec:    # expert iteration: soft target = softmax(sim estimate / TAU) over the candidate set
                    keyidx = {}
                    for k, m in enumerate(it["_moves"]): keyidx.setdefault((m[1], m[2], m[3], m[5]), k)
                    soft = np.zeros(len(it["_moves"]), dtype=np.float32); es = np.array(rec["cand_est"], dtype=np.float64)
                    w = np.exp((es - es.max()) / self.tau)
                    srt = np.sort(es)[::-1]; it["margin"] = np.float32(srt[0] - srt[1] if len(srt) > 1 else 99.0)
                    for (r_, c_, d_, L_, w_), ww in zip(rec["cands"], w):
                        k = keyidx.get((r_, c_, d_, w_))
                        if k is not None: soft[k] += ww
                    if soft.sum() > 0: it["soft"] = soft / soft.sum()
                    else: continue
                if "v_base" in rec:   # policy gradient: advantage = outcome - baseline (spread/100); PPO needs the behaviour log-prob
                    it["adv"] = np.float32(final / 100 - rec["v_base"]); it["logp_old"] = np.float32(rec.get("logp_old", 0.0))
                # k-ply target: spread change (my perspective) over the next kply half-moves; final spread if the game ends first
                tk = min(ti + self.kply, len(traj) - 1); r2 = traj[tk]
                sp_now = rec["scores"][0] - rec["scores"][1]
                if tk == len(traj) - 1 or ti + self.kply >= len(traj): sp_k = final
                else: sp_k = (r2["scores"][0] - r2["scores"][1]) * (1 if r2["mover"] == rec["mover"] else -1)
                it["value_k"] = np.float32((sp_k - sp_now) / 100)
                return it

def collate_moves(items):
    B = len(items)
    out = {k: torch.tensor(np.stack([it[k] for it in items])) for k in ("board", "rack", "unseen", "glob")}
    if "widx" in items[0]:
        M = max(len(it["widx"]) for it in items)
        if "soft" in items[0]:
            soft = torch.zeros(B, M)
            for b, it in enumerate(items): soft[b, :len(it["soft"])] = torch.tensor(it["soft"])
            out["soft"] = soft; out["margin"] = torch.tensor([it.get("margin", 99.0) for it in items])
        mf = torch.zeros(B, M, MF); leave = torch.zeros(B, M, 27); widx = torch.zeros(B, M, dtype=torch.long); valid = torch.zeros(B, M, dtype=torch.bool)
        for b, it in enumerate(items):
            n = len(it["widx"]); mf[b, :n] = torch.tensor(it["mf"]); leave[b, :n] = torch.tensor(it["leave"]); widx[b, :n] = torch.tensor(it["widx"]); valid[b, :n] = True
        out.update(mf=mf, leave=leave, widx=widx, valid=valid, target=torch.tensor([it["target"] for it in items]), value=torch.tensor([it["value"] for it in items]),
                   w_pol=torch.tensor([it.get("w_pol", 1.0) for it in items]), value_k=torch.tensor([it.get("value_k", 0.0) for it in items]),
                   adv=torch.tensor([it.get("adv", 0.0) for it in items]), logp_old=torch.tensor([it.get("logp_old", 0.0) for it in items]))
    if "rmask" in items[0]:
        for li in range(len(RET_LENS)):
            C = max(len(it["rmask"][li]) for it in items); rm = torch.zeros(B, C, dtype=torch.long); rv = torch.zeros(B, C, dtype=torch.bool)
            for b, it in enumerate(items):
                n = len(it["rmask"][li]); rm[b, :n] = torch.tensor(it["rmask"][li]); rv[b, :n] = True
            out[f"rmask{li}"] = rm; out[f"rvalid{li}"] = rv
    return out

class PolicyNet(nn.Module):
    def __init__(self, lex, D=256, layers=3, use_lex=False):
        super().__init__()
        self.D, self.use_lex = D, use_lex
        self.wenc = WordEncoder(lex, D, slot=False)
        self.bemb = nn.Embedding(27, 32)
        self.cnn = nn.Sequential(nn.Conv2d(32 + 5, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 16, 1), nn.GELU(), nn.Flatten(), nn.Linear(16 * 225, D))
        self.register_buffer("prem", F.one_hot(torch.tensor(PREM), 5).permute(2, 0, 1).float()[None])
        self.rack = nn.Linear(27, D); self.unseen = nn.Linear(27, D); self.glob = nn.Linear(3, D)
        self.mfeat = nn.Linear(MF + 27, D); self.remb = nn.Embedding(15, D); self.cemb = nn.Embedding(15, D)
        self.ttype = nn.Parameter(torch.randn(6, D) * 0.02); self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        enc = nn.TransformerEncoderLayer(D, 4, D * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(D)
        self.pol = nn.Linear(D, 1); self.val = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        if use_lex:
            self.qenc = QueryEncoder(D, 3)
            self.register_buffer("keys", torch.zeros(lex.NW, D)); self.register_buffer("Wt", torch.tensor(lex.W, dtype=torch.float32))
            self.rfeat = nn.Linear(2, D); self.lane_tpl = {}
            for L in RET_LENS:      # an all-empty, unconstrained lane of length L as the query's lane tokens
                x = np.zeros((MAXL, LANE_F), dtype=np.float32); x[:L, 26] = 1; x[:L, 27:53] = 1; x[:L, 54] = 1; x[:L, 59] = 1
                self.register_buffer(f"lane{L}", torch.tensor(x)[None])
    @torch.no_grad()
    def refresh_keys(self, chunk=8192):
        for i in range(0, self.keys.shape[0], chunk):
            idx = torch.arange(i, min(i + chunk, self.keys.shape[0]), device=self.keys.device); self.keys[i:i + chunk] = self.wenc(idx).float()
    def load_retrieval(self, ckpt):
        sd = torch.load(ckpt, map_location=self.keys.device)
        self.wenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("wenc.") and k[5:].split(".")[0] in ("emb", "pos", "tr", "out")}, strict=False)
        self.qenc.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("qenc.")}); self.refresh_keys()
    def forward(self, b, with_moves=True):
        B = b["rack"].shape[0]; M = b["widx"].shape[1] if with_moves else 0
        board = self.bemb(b["board"]).permute(0, 3, 1, 2); board = torch.cat([board, self.prem.expand(B, -1, -1, -1)], 1)
        pos = torch.stack([self.cnn(board) + self.ttype[0], self.rack(b["rack"] / 3) + self.ttype[1], self.unseen(b["unseen"] / 12) + self.ttype[2], self.glob(b["glob"]) + self.ttype[3]], 1)
        dev_ = pos.device
        toks = [self.cls.expand(B, -1, -1), pos]; pads = [torch.zeros(B, 5, dtype=torch.bool, device=dev_)]
        if self.use_lex:
            for li, L in enumerate(RET_LENS):
                lane = getattr(self, f"lane{L}").expand(B, -1, -1)
                glob5 = torch.cat([b["glob"][:, :2], torch.full((B, 1), L / 15, device=dev_), torch.zeros(B, 2, device=dev_)], 1)
                q = self.qenc(b["rack"], b["unseen"], glob5, lane, b["board"], torch.full((B,), L, device=dev_, dtype=torch.long))
                rm, rv = b[f"rmask{li}"], b[f"rvalid{li}"]
                logits = torch.einsum("bd,bcd->bc", q, self.keys[rm]) * (self.D ** -0.5); logits = logits.masked_fill(~rv, -1e4)
                k = min(KRET, rm.shape[1]); top = logits.topk(k, 1); idx = rm.gather(1, top.indices)
                need = (self.Wt[idx] - b["rack"][:, None, :26]).clamp_min(0).sum(-1)      # letters still needed from the bag
                rt = self.keys[idx] + self.rfeat(torch.stack([top.values, need / 3], -1)) + self.ttype[5]
                toks.append(rt); pads.append(~rv.gather(1, top.indices))
        if with_moves:
            wk = self.wenc(b["widx"])
            mv = wk + self.mfeat(torch.cat([b["mf"], b["leave"] / 3], -1)) + self.remb((b["mf"][..., 2] * 14).round().long()) + self.cemb((b["mf"][..., 3] * 14).round().long()) + self.ttype[4]
            toks.append(mv); pads.append(~b["valid"])
        x = torch.cat(toks, 1); pad = torch.cat(pads, 1)
        h = self.norm(self.tr(x, src_key_padding_mask=pad))
        if not with_moves: return None, self.val(h[:, 0]).squeeze(-1)
        plog = self.pol(h[:, -M:]).squeeze(-1).masked_fill(~b["valid"], -1e4)
        return plog, self.val(h[:, 0]).squeeze(-1)
