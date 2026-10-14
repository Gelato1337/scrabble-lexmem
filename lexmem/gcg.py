"""Human tournament games (cross-tables self-annotated GCG) -> positions in our trajectory format (same keys as replay.py).
Self-annotated games give the annotator's full rack every turn; the opponent's line usually shows only the tiles played.
A player's racks are 'known' if >= 80 % of their pre-endgame racks have 7 tiles (both are, for Woogles exports).
Positions where the mover's rack is unknown are kept to advance the board but marked known=False (no policy target).
Every placement is re-scored with our engine; the game is cut at the first mismatch or inconsistency.
Phonies withdrawn (`--`) become a pass by that player. Scores follow the GCG cumulative column (time penalties,
challenge bonuses and end-rack adjustments included)."""
import re, numpy as np
from lexmem.core import *
from lexmem.replay import parse_coord, score_placement, place, rack27

LINE = re.compile(r"^>([^:]+):\s*(.*)$")
PCOORD = re.compile(r"^(\d+[A-O]|[A-O]\d+)$")

def parse_lines(text):
    """-> (events, meta). event = dict(player, rack or None, kind in play/exch/pass/withdraw/adjust, coord, word, score, cum)."""
    ev = []; meta = {}
    for ln in text.splitlines():
        if ln.startswith("#"):
            k, _, v = ln[1:].partition(" "); meta.setdefault(k.strip(), v.strip()); continue
        m = LINE.match(ln)
        if not m: continue
        toks = m.group(2).split()
        if len(toks) < 2 or not re.fullmatch(r"-?\d+", toks[-1]) or not re.fullmatch(r"[+-]-?\d+", toks[-2]): return None, meta
        cum, score, rest = int(toks[-1]), int(toks[-2].replace("+-", "-")), toks[:-2]
        e = dict(player=m.group(1).strip(), cum=cum, score=score, rack=None)
        if len(rest) >= 2 and re.fullmatch(r"[A-Z?]+", rest[0]):
            e["rack"] = rest[0] if len(rest[0]) <= 7 else None; rest = rest[1:]    # >7 tiles = annotator typo: rack unknown
        if len(rest) == 2 and re.fullmatch(r"\d+[a-o]|[a-o]\d+", rest[0]): rest[0] = rest[0].upper()
        if len(rest) == 2 and PCOORD.match(rest[0]):
            e.update(kind="play", coord=rest[0], word=rest[1])
        elif len(rest) == 1 and rest[0] == "--": e["kind"] = "withdraw"
        elif len(rest) == 1 and rest[0] == "-": e["kind"] = "pass"
        elif len(rest) == 1 and re.fullmatch(r"-[A-Z?]+", rest[0]): e.update(kind="exch", tiles=rest[0][1:])
        elif len(rest) == 1 and re.fullmatch(r"-\d+", rest[0]): e.update(kind="exch", tiles=None)   # counted exchange, tiles hidden
        elif len(rest) == 1 and rest[0].startswith("("): e["kind"] = "adjust"
        elif len(rest) == 0 and e["rack"] is None: e["kind"] = "adjust"
        else: return None, meta
        ev.append(e)
    return ev, meta

def to_letters(word):
    """GCG word -> our letters: '(E)' and '.' are through-tiles ('.'), lowercase = blank."""
    return re.sub(r"\(([A-Za-z]+)\)", lambda m: "." * len(m.group(1)), word)

def replay_gcg(lex, text):
    """-> (traj, final_scores, stats) or (None, None, reason)."""
    ev, meta = parse_lines(text)
    if ev is None: return None, None, "unparsed"
    if meta.get("game-type", "classic") != "classic": return None, None, "variant"
    if meta.get("lexicon", "NWL")[:3] not in ("NWL", "TWL", "OTC"): return None, None, "lexicon"
    players = []
    for e in ev:
        if e["player"] not in players: players.append(e["player"])
    if len(players) != 2: return None, None, "players"
    # which players' racks are known: share of 7-tile racks on play/exch/pass lines
    full = {p: [0, 0] for p in players}
    for e in ev:
        if e["kind"] in ("play", "exch", "pass"):
            full[e["player"]][1] += 1; full[e["player"]][0] += int(e["rack"] is not None and len(e["rack"]) == 7)
    known = {p: full[p][1] > 0 and full[p][0] / full[p][1] >= 0.8 for p in players}
    if not any(known.values()): return None, None, "no-racks"
    board, blanks = empty_board(); scores = {p: 0 for p in players}; traj = []; stats = dict(plays=0, cut=None)
    for e in ev:
        p = e["player"]; me = players.index(p); opp = players[1 - me]
        if e["kind"] == "adjust": scores[p] = e["cum"]; continue
        if e["kind"] == "withdraw":
            # the phony was this player's last event: restore the board and turn it into a pass
            last = next((t for t in reversed(traj) if t["mover"] == me), None)
            if last is None or last["move"][0] != "play": stats["cut"] = "withdraw-orphan"; break
            board, blanks = last["board"].copy(), last["blanks"].copy()
            last.update(move=("pass",), placed=[], phony=True); scores[p] = e["cum"]; continue
        rk = known[p] and e["rack"] is not None
        rack = rack27(e["rack"]) if rk else np.zeros(27, dtype=np.int16)
        on_board = counts27(list(board[board >= 0])); on_board[BLANK] = int(blanks.sum())
        for (rr, cc2) in zip(*np.where(blanks)): on_board[board[rr, cc2]] -= 1
        unseen = FULL_BAG - on_board - rack
        if (unseen < 0).any(): stats["cut"] = "rack-inconsistent"; break
        bag = max(0, int(unseen.sum()) - 7)
        rec = dict(board=board.copy(), blanks=blanks.copy(), rack=rack.copy(), unseen=unseen.astype(np.int16),
                   scores=(scores[p], scores[opp]), bag=bag, turn=len(traj), mover=me, known=rk, drawn=[])
        if e["kind"] == "play":
            pc = parse_coord(e["coord"] + " " + to_letters(e["word"]))
            if pc is None: stats["cut"] = "coord"; break
            r, c, d, letters = pc
            if not re.fullmatch(r"[A-Za-z.]+", letters): stats["cut"] = "bad-word"; break
            dr, dc = ((0, 1), (1, 0))[d]
            if not all(0 <= r + dr * j < N and 0 <= c + dc * j < N for j in range(len(letters))): stats["cut"] = "off-board"; break
            # old-style GCG writes the through-tiles as letters: turn occupied squares that match into '.'
            fixed = ""
            for j, ch in enumerate(letters):
                b = board[r + dr * j, c + dc * j]
                if ch != "." and b >= 0:
                    if LET[b] != ch.upper(): fixed = None; break
                    fixed += "."
                else: fixed += ch
            if fixed is None or all(ch == "." for ch in fixed): stats["cut"] = "occupied"; break
            letters = fixed
            if rk:
                need = counts27([BLANK if ch.islower() else IDX[ch] for ch in letters if ch != "."])
                if ((rack - need) < 0).any(): stats["cut"] = "rack-missing-tiles"; break
            cc = cross_checks(lex, board, blanks)
            try: s, _ = score_placement(lex, board, blanks, cc, r, c, d, letters)
            except AssertionError: stats["cut"] = "placement"; break
            if s != e["score"]: stats["cut"] = f"score {s}!={e['score']}"; break
            stats["plays"] += 1
            board, blanks, placed = place(board, blanks, r, c, d, letters)
            rec.update(move=("play", r, c, d, len(letters), letters, s), placed=placed)
        elif e["kind"] == "exch": rec.update(move=("exchange", e.get("tiles")), placed=[])
        else: rec.update(move=("pass",), placed=[])
        scores[p] = e["cum"]; traj.append(rec)
    if not traj: return None, None, stats["cut"] or "empty"
    if stats["cut"]: return traj, None, stats   # truncated: no trustworthy final score
    return traj, (scores[players[0]], scores[players[1]]), stats
