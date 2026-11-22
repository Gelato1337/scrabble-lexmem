"""Rack-inference positions (round 6) from HastyBot self-play games (data/games_<SRC>.pkl, lexmem/replay.py records: every
record holds the mover's rack). One position per play of HastyBot ("the opponent") with tiles left in the bag before the move
and at least one tile kept; the observer is the other player, whose rack is the next record's rack.
  P     = the tiles the opponent placed (known to the observer; blank = index 26)
  U     = the tiles unseen to the observer before the move = bag + the opponent's rack
  leave = the opponent's rack minus P (the target: what it kept)
Stored as arrays in data/infer_<OUT>.npz: board, blanks (before the move), P, U, leave, scores (opponent, observer), bag, turn,
move (r, c, d, L, widx, score), gi; letters = the move's letters ('.' = through tile, lowercase = blank).
Usage: PYTHONPATH=. python prep/infer_positions.py SRC OUT     e.g. hasty_train train, hasty_test test"""
import sys, pickle, numpy as np
from lexmem.core import Lexicon, LET, IDX, BLANK, FULL_BAG, N
src, out = sys.argv[1], sys.argv[2]
lex = Lexicon("data/NWL23.txt"); games = pickle.load(open(f"data/games_{src}.pkl", "rb"))
cols = {k: [] for k in ("board", "blanks", "P", "U", "leave", "scores", "bag", "turn", "move", "gi", "letters")}
skip = dict(kind=0, bag=0, mover=0, leave=0, unseen=0, word=0)
for gi, (tr, _) in enumerate(games):
    for ti in range(len(tr) - 1):
        rec, nxt = tr[ti], tr[ti + 1]; m = rec["move"]
        if m[0] != "play": skip["kind"] += 1; continue
        if rec["bag"] < 1: skip["bag"] += 1; continue
        if nxt["mover"] == rec["mover"]: skip["mover"] += 1; continue
        _, r0, c0, d0, L0, letters, s = m; dr, dc = ((0, 1), (1, 0))[d0]
        P = np.zeros(27, dtype=np.int16)
        for ch in letters:
            if ch != ".": P[BLANK if ch.islower() else IDX[ch]] += 1
        leave = rec["rack"].astype(np.int16) - P
        if (leave < 0).any() or leave.sum() == 0: skip["leave"] += 1; continue
        U = rec["unseen"].astype(np.int16) + rec["rack"].astype(np.int16) - nxt["rack"].astype(np.int16)
        t = np.where(rec["blanks"], BLANK, rec["board"].astype(np.int64))[rec["board"] >= 0]
        on_board = np.bincount(t, minlength=27).astype(np.int16)
        if (U < 0).any() or (U - rec["rack"] < 0).any() or not np.array_equal(U, FULL_BAG - on_board - nxt["rack"]): skip["unseen"] += 1; continue
        word = "".join(LET[rec["board"][r0 + dr * j, c0 + dc * j]] if ch == "." else ch.upper() for j, ch in enumerate(letters))
        w = lex.WIDX.get(word)
        if w is None: skip["word"] += 1; continue
        cols["board"].append(rec["board"].astype(np.int8)); cols["blanks"].append(rec["blanks"].astype(bool)); cols["P"].append(P); cols["U"].append(U)
        cols["leave"].append(leave); cols["scores"].append(np.array(rec["scores"], dtype=np.int16)); cols["bag"].append(rec["bag"]); cols["turn"].append(rec["turn"])
        cols["move"].append(np.array([r0, c0, d0, L0, w, s], dtype=np.int32)); cols["gi"].append(gi); cols["letters"].append(letters)
arr = {k: np.array(v) for k, v in cols.items()}
np.savez(f"data/infer_{out}.npz", **arr)
print(f"{len(games)} games -> {len(arr['gi'])} positions in data/infer_{out}.npz; skipped {skip}; kept tiles per position {arr['leave'].sum(1).mean():.2f}, bag {arr['bag'].mean():.1f}")
