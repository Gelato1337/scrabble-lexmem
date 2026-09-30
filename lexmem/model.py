import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from lexmem.data import LANE_F, MAXL

class WordEncoder(nn.Module):
    """letter-composition encoder (small transformer over letters) + per-word slot."""
    def __init__(self, lex, D=256, dl=128, layers=2, slot=True):
        super().__init__()
        self.register_buffer("POS", torch.tensor(lex.POS))        # (NW,15) 26=pad
        self.register_buffer("LEN", torch.tensor(lex.LEN))
        self.emb = nn.Embedding(27, dl); self.pos = nn.Parameter(torch.randn(MAXL, dl) * 0.02)
        enc = nn.TransformerEncoderLayer(dl, 4, dl * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dl, D)
        self.slot = nn.Embedding(lex.NW, D) if slot else None
        if slot: nn.init.normal_(self.slot.weight, std=0.02)
    def forward(self, idx):
        shape = idx.shape; idx = idx.reshape(-1)
        p = self.POS[idx]; pad = p == 26
        x = self.emb(p) + self.pos
        h = self.tr(x, src_key_padding_mask=pad)
        h = (h * (~pad)[..., None]).sum(1) / (~pad).sum(1, keepdim=True)
        k = self.out(h)
        if self.slot is not None: k = k + self.slot(idx)
        return k.reshape(*shape, -1)

class QueryEncoder(nn.Module):
    """tokens: [CLS, rack, unseen, glob, board-summary, lane positions x15] -> transformer -> query."""
    def __init__(self, D=256, layers=3, heads=4):
        super().__init__()
        self.rack = nn.Linear(27, D); self.unseen = nn.Linear(27, D); self.glob = nn.Linear(5, D)
        self.board_emb = nn.Embedding(27, 8); self.board = nn.Linear(225 * 8, D)
        self.lane = nn.Linear(LANE_F, D); self.lanepos = nn.Parameter(torch.randn(MAXL, D) * 0.02)
        self.ttype = nn.Parameter(torch.randn(6, D) * 0.02); self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        enc = nn.TransformerEncoderLayer(D, heads, D * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(D)
    def forward(self, rack, unseen, glob, lane, board, L):
        B = rack.shape[0]
        toks = [self.cls.expand(B, -1, -1),
                (self.rack(rack / 3) + self.ttype[0])[:, None],
                (self.unseen(unseen / 12) + self.ttype[1])[:, None],
                (self.glob(glob) + self.ttype[2])[:, None],
                (self.board(self.board_emb(board).flatten(1)) + self.ttype[3])[:, None],
                self.lane(lane) + self.lanepos + self.ttype[4]]
        x = torch.cat(toks, 1)
        pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)
        pad[:, 5:] = torch.arange(MAXL, device=x.device)[None] >= L[:, None]
        h = self.tr(x, src_key_padding_mask=pad)
        return self.norm(h[:, 0])

class LexMem(nn.Module):
    def __init__(self, lex, D=256, slot=True, qlayers=3):
        super().__init__()
        self.D = D; self.wenc = WordEncoder(lex, D, slot=slot); self.qenc = QueryEncoder(D, qlayers)
        self.scale = nn.Parameter(torch.tensor(float(D) ** -0.5))
    def forward(self, batch, keys=None):
        q = self.qenc(batch["rack"], batch["unseen"], batch["glob"], batch["lane"], batch["board"], batch["L"])
        k = self.wenc(batch["cand"]) if keys is None else keys[batch["cand"]]      # (B,C,D)
        logits = torch.einsum("bd,bcd->bc", q, k) * self.scale
        return logits.masked_fill(~batch["valid"], -1e4)
