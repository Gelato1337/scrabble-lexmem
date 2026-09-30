"""Replay Macondo autoplay logs ({experiment}.txt + games-{experiment}.txt) into positions in our format.
Every placement is re-scored with our engine and checked against the logged score."""
import re, csv, numpy as np
from collections import defaultdict
from lexmem.core import *

COORD = re.compile(r"^\s*(?:(\d+)([A-O])|([A-O])(\d+))\s+(\S+)\s*$")

def parse_coord(play):
    """returns (r, c, d, letters) where letters is the word with '.' for through tiles, lowercase = blank."""
    m = COORD.match(play)
    if not m: return None
    if m.group(1): r, c, d = int(m.group(1)) - 1, ord(m.group(2)) - 65, 0
    else: r, c, d = int(m.group(4)) - 1, ord(m.group(3)) - 65, 1
    return r, c, d, m.group(5)

def score_placement(lex, board, blanks, cc, r, c, d, letters):
    """exact score of placing `letters` (with '.' through-tiles, lowercase blanks) at (r,c,d)."""
    L = len(letters); info = lane_info(board, blanks, cc, r, c, d, L)
    dr, dc = ((0, 1), (1, 0))[d]
    main = 0; wmul = 1; cross = 0; placed = 0
    for j, ch in enumerate(letters):
        rr, c2 = r + dr * j, c + dc * j
        if ch == ".":
            assert board[rr, c2] >= 0, "through-tile on empty square"
            main += 0 if blanks[rr, c2] else VAL[board[rr, c2]]
            continue
        assert board[rr, c2] < 0, "placing on occupied square"
        val = 0 if ch.islower() else VAL[IDX[ch]]
        main += val * LMULT[rr, c2]; wmul *= WMULT[rr, c2]; placed += 1
        if info["has"][j]: cross += (info["base"][j] + val * LMULT[rr, c2]) * WMULT[rr, c2]
    return main * wmul + cross + (50 if placed == 7 else 0), placed

def place(board, blanks, r, c, d, letters):
    board = board.copy(); blanks = blanks.copy(); dr, dc = ((0, 1), (1, 0))[d]; placed = []
    for j, ch in enumerate(letters):
        if ch == ".": continue
        rr, c2 = r + dr * j, c + dc * j
        board[rr, c2] = IDX[ch.upper()]; blanks[rr, c2] = ch.islower(); placed.append((rr, c2, IDX[ch.upper()]))
    return board, blanks, placed

def rack27(s):
    return counts27([BLANK if ch == "?" else IDX[ch] for ch in s])

def replay_log(lex, turns_path, games_path, max_games=None, verbose=False):
    first = {}
    for row in csv.DictReader(open(games_path)):
        first[row["gameID"]] = row["first"]
    names = None
    by_game = defaultdict(list)
    with open(turns_path) as f:
        rd = csv.DictReader(f); names = rd.fieldnames
        for row in rd: by_game[row["gameID"]].append(row)
    games = []; nbad = 0; nplays = 0
    for gid, rows in by_game.items():
        if gid not in first: continue
        if max_games and len(games) >= max_games: break
        # who is p1? the games file names players HastyBot (p1) / HastyBot1 (p2); "first" gives the name of the first mover
        p_first = "p1" if not first[gid].endswith("1") else "p2"
        order = {p_first: 0, ("p2" if p_first == "p1" else "p1"): 1}
        rows.sort(key=lambda x: (int(x["turn"]), order[x["playerID"]]))
        board, blanks = empty_board(); scores = [0, 0]; traj = []; ok = True
        for i, row in enumerate(rows):
            me = order[row["playerID"]]; rack = rack27(row["rack"])
            on_board = counts27(list(board[board >= 0])) ; on_board[BLANK] = int(blanks.sum())
            for (rr, cc2) in zip(*np.where(blanks)): on_board[board[rr, cc2]] -= 1
            unseen = FULL_BAG - on_board - rack
            if (unseen < 0).any(): ok = False; break
            bag = int(unseen.sum()) - 7 if int(unseen.sum()) >= 7 else 0
            rec = dict(board=board.copy(), blanks=blanks.copy(), rack=rack.copy(), unseen=unseen.astype(np.int16),
                       scores=(scores[me], scores[1 - me]), bag=bag, turn=i, mover=me, hasty_equity=float(row["equity"]))
            play = row["play"].strip(); pc = parse_coord(play)
            if pc:
                r, c, d, letters = pc
                cc = cross_checks(lex, board, blanks)
                s, placed_n = score_placement(lex, board, blanks, cc, r, c, d, letters)
                nplays += 1
                if s != int(row["score"]):
                    nbad += 1
                    if verbose: print("score mismatch", gid, play, s, row["score"])
                board, blanks, placed = place(board, blanks, r, c, d, letters)
                scores[me] += int(row["score"])
                rec.update(move=("play", r, c, d, len(letters), letters, int(row["score"])), placed=placed)
            elif play.lower().startswith("(exch"):
                rec.update(move=("exchange", play), placed=[])
            else:
                rec.update(move=("pass",), placed=[])
            # drawn = next rack of this player minus leave
            leave = rack27(row["leave"])
            nxt = next((x for x in rows[i + 1:] if x["playerID"] == row["playerID"]), None)
            drawn = (rack27(nxt["rack"]) - leave) if nxt else np.zeros(27, dtype=np.int16)
            rec["drawn"] = [t for t in range(27) for _ in range(max(0, int(drawn[t])))]
            traj.append(rec)
        if ok and traj: games.append((traj, tuple(scores)))
    return games, nplays, nbad

if __name__ == "__main__":
    import sys, pickle, time, glob
    lex = Lexicon(sys.argv[1]); d = sys.argv[2]; out = sys.argv[3]
    tp = [p for p in glob.glob(d + "/*.txt") if "/games-" not in p][0]; gp = glob.glob(d + "/games-*.txt")[0]
    t = time.time(); games, nplays, nbad = replay_log(lex, tp, gp, verbose=True)
    print(f"{len(games)} games, {sum(len(g[0]) for g in games)} positions, {nplays} plays, {nbad} score mismatches, {time.time()-t:.0f}s")
    pickle.dump(games, open(out, "wb"))
