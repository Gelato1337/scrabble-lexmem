"""
Play games: our policy (Python engine, NWL23) vs HastyBot (Macondo shell as a subprocess).
Game state is kept in Python; HastyBot receives the position as CGP and answers with its top-equity play.
Usage: MACONDO=~/opt/macondo python harness.py --games 4 --policy greedy
"""
import os, sys, re, subprocess, time, random, argparse, numpy as np
from lexmem.core import *
from lexmem.replay import parse_coord, score_placement, place

MACONDO = os.environ.get("MACONDO", os.path.expanduser("~/opt/macondo"))
ROW = re.compile(r"^\s*1:\s+(.*?)\s{2,}")

class HastyBot:
    def __init__(self, lexicon="NWL23"):
        self.p = subprocess.Popen([os.path.join(MACONDO, "bin/shell")], cwd=MACONDO, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  env=dict(os.environ, GOMAXPROCS=os.environ.get("MACONDO_THREADS", "2")))
        self.lexicon = lexicon
    def cgp(self, board, blanks, rack27, scores, zeros):
        rows = []
        for r in range(N):
            s = ""; run = 0
            for c in range(N):
                if board[r, c] < 0: run += 1
                else:
                    if run: s += str(run); run = 0
                    ch = LET[board[r, c]]; s += ch.lower() if blanks[r, c] else ch
            if run: s += str(run)
            rows.append(s)
        rack = "".join("?" if t == BLANK else LET[t] for t in range(27) for _ in range(rack27[t]))
        return f"{'/'.join(rows)} {rack}/ {scores[0]}/{scores[1]} {zeros} lex {self.lexicon};"
    def move(self, board, blanks, rack27, scores, zeros):
        self.p.stdin.write(f"load cgp {self.cgp(board, blanks, rack27, scores, zeros)}\ngen 1\n"); self.p.stdin.flush()
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("macondo died")
            m = ROW.match(line)
            if m: return m.group(1).strip()
            if "Equity" in line and "Move" in line:   # header; the row follows (or nothing if no moves)
                nxt = self.p.stdout.readline(); m = ROW.match(nxt)
                return m.group(1).strip() if m else "(Pass)"
            if time.time() - t0 > 300:
                print("WARNING: macondo timeout in HastyBot.move -> pass", flush=True); return "(Pass)"
    def close(self):
        try: self.p.stdin.write("exit\n"); self.p.stdin.flush(); self.p.wait(timeout=5)
        except Exception: self.p.kill()

def macondo_play_to_letters(play):
    """'G7 M(E)GAHITS' -> (r,c,d,'M.GAHITS'); '(exch ABC)' / '(Pass)' passthrough."""
    if play.startswith("("): return play
    parts = play.split()
    if len(parts) < 2: return None
    coord, word = parts[0], parts[1]          # a long play overflows the column and the leave follows after one space
    letters = re.sub(r"\(([A-Za-z]+)\)", lambda m: "." * len(m.group(1)), word)
    pc = parse_coord(coord + " " + letters); return pc

def greedy_policy(lex, board, blanks, cc, rack, unseen, rng, temp=0.0):
    mv = legal_moves(lex, board, blanks, cc, rack)
    if not mv: return None
    if temp <= 0: return max(mv)
    sc = np.array([m[0] for m in mv], dtype=float); p = np.exp((sc - sc.max()) / temp); p /= p.sum()
    return mv[rng.choices(range(len(mv)), weights=p)[0]]

def play_vs_hasty(lex, policy, hasty, seed, we_first):
    rng = random.Random(seed)
    bag = [i for i in range(27) for _ in range(FULL_BAG[i])]; rng.shuffle(bag)
    board, blanks = empty_board(); racks = [counts27(draw_tiles(bag, 7, rng)), counts27(draw_tiles(bag, 7, rng))]
    scores = [0, 0]; zeros = 0; turn = 0; log = []
    us = 0 if we_first else 1
    while True:
        me = turn % 2; rack = racks[me]
        if me == us:
            cc = cross_checks(lex, board, blanks)
            mv = policy(lex, board, blanks, cc, rack, counts27(bag) + racks[1 - me], rng)
            if mv is None: action = "(Pass)"
            else:
                s, r, c, d, L, widx = mv
                letters = ""; dr, dc = ((0, 1), (1, 0))[d]
                for j in range(L):
                    rr, c2 = r + dr * j, c + dc * j
                    letters += "." if board[rr, c2] >= 0 else LET[lex.POS[widx, j]]
                # blanks: letters not on rack -> lowercase
                need = counts27([IDX[ch] for ch in letters if ch != "."]); tmp = rack.copy(); out = ""
                for ch in letters:
                    if ch == ".": out += "."; continue
                    if tmp[IDX[ch]] > 0: tmp[IDX[ch]] -= 1; out += ch
                    else: tmp[BLANK] -= 1; out += ch.lower()
                action = (r, c, d, out)
        else:
            play = hasty.move(board, blanks, rack, (scores[me], scores[1 - me]), zeros)
            action = macondo_play_to_letters(play)
            if action is None:
                print(f"UNPARSED hasty play {play!r} -> treated as pass", flush=True); action = "(Pass)"
        if isinstance(action, tuple):
            r, c, d, letters = action
            cc = cross_checks(lex, board, blanks)
            bad = None
            if me != us:   # validate HastyBot's play under our rules: main word + every cross word must be in our lexicon
                dr, dc = ((0, 1), (1, 0))[d]
                if not all(0 <= r + dr * j < N and 0 <= c + dc * j < N for j in range(len(letters))): bad = "off-board"
                else:
                    main = "".join((LET[board[r+dr*j, c+dc*j]] if ch == "." else ch.upper()) for j, ch in enumerate(letters))
                    if main not in lex.WSET: bad = f"main word {main} not in lexicon"
                    for j, ch in enumerate(letters):
                        if ch != "." and (board[r+dr*j, c+dc*j] >= 0 or not cc[d][0][r+dr*j, c+dc*j][IDX[ch.upper()]]): bad = f"cross-check/occupied at {j}"
            if bad:
                print(f"INVALID hasty play {play!r} ({bad}) -> treated as pass; cgp={hasty.cgp(board, blanks, rack, (scores[me], scores[1-me]), zeros)}", flush=True)
                zeros += 1; log.append((turn, me, "pass", 0)); racks[me] = rack; turn += 1
                if zeros >= 6 or turn > 80: break
                continue
            s, nplaced = score_placement(lex, board, blanks, cc, r, c, d, letters)
            board, blanks, placed = place(board, blanks, r, c, d, letters)
            for ch in letters:
                if ch == ".": continue
                t = BLANK if ch.islower() else IDX[ch]; rack[t] -= 1
            assert (rack >= 0).all(), (letters, rack)
            rack += counts27(draw_tiles(bag, 7 - int(rack.sum()), rng)); scores[me] += s; zeros = 0
            log.append((turn, me, letters, s))
        elif action.lower().startswith("(exch"):
            tiles = action[6:-1].strip(); ex = [BLANK if ch == "?" else IDX[ch.upper()] for ch in tiles]
            for t in ex: rack[t] -= 1
            newt = draw_tiles(bag, len(ex), rng); bag.extend(ex); rng.shuffle(bag); rack += counts27(newt); zeros += 1
            log.append((turn, me, action, 0))
        else:
            zeros += 1; log.append((turn, me, "pass", 0))
        racks[me] = rack; turn += 1
        if rack.sum() == 0 and len(bag) == 0:
            pen = int((racks[1 - me] * VAL[:27]).sum()); scores[me] += pen; scores[1 - me] -= pen; break
        if zeros >= 6 or turn > 80: break
    return scores[us] - scores[1 - us], scores, log

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--games", type=int, default=4); ap.add_argument("--policy", default="greedy")
    ap.add_argument("--lexicon", default="data/NWL23.txt"); ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lex = Lexicon(a.lexicon); hasty = HastyBot()
    pol = (lambda *x: greedy_policy(*x, temp=a.temp)) if a.policy == "greedy" else None
    res = []
    t0 = time.time()
    for g in range(a.games):
        spread, scores, log = play_vs_hasty(lex, pol, hasty, a.seed + g, we_first=(g % 2 == 0))
        res.append(spread); print(f"game {g}: us {'first' if g%2==0 else 'second'} spread {spread:+d} scores {scores} turns {len(log)} t={time.time()-t0:.0f}s", flush=True)
    res = np.array(res); print(f"\n{a.policy} vs HastyBot: {a.games} games, win rate {np.mean(res > 0):.3f} (ties {np.mean(res==0):.2f}), mean spread {res.mean():+.1f} ± {res.std()/np.sqrt(len(res)):.1f}")
    hasty.close()
