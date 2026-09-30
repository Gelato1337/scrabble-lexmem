"""Macondo shell as a move-generation server: load a CGP position, `gen N`, parse the equity-sorted move list
into our move tuples (score, r, c, d, L, widx). Blank plays: lowercase letters in the word."""
import os, re, subprocess, time
from lexmem.core import N, LET, IDX, BLANK
from lexmem.replay import parse_coord
MACONDO = os.environ.get("MACONDO", os.path.expanduser("~/opt/macondo"))
ROW = re.compile(r"^\s*(\d+):\s+(\S+)\s+(\S+)")          # idx, coord, word  (leave/score/equity follow)
EXCH = re.compile(r"^\s*(\d+):\s+\((exch|Pass)")

class MacondoGen:
    def __init__(self, lexicon="NWL23"):
        self.p = subprocess.Popen([os.path.join(MACONDO, "bin/shell")], cwd=MACONDO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  env=dict(os.environ, GOMAXPROCS=os.environ.get("MACONDO_THREADS", "2")))
        self.lexicon = lexicon
    @staticmethod
    def cgp(board, blanks, rack27, scores, zeros, lexicon):
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
        return f"{'/'.join(rows)} {rack}/ {scores[0]}/{scores[1]} {zeros} lex {lexicon};"
    def gen(self, lex, board, blanks, rack27, scores=(0, 0), zeros=0, n=400):
        """returns list of (score, r, c, d, L, widx, letters, equity) sorted by Macondo equity; exchanges/passes skipped."""
        self.p.stdin.write(f"load cgp {self.cgp(board, blanks, rack27, scores, zeros, self.lexicon)}\ngen {n}\ngid\n"); self.p.stdin.flush()
        out = []; seen_header = False
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("macondo died")
            if "Equity" in line and "Move" in line: seen_header = True; continue
            if not seen_header: continue
            if EXCH.match(line): continue
            m = ROW.match(line)
            if m:
                coord, word = m.group(2), m.group(3)
                letters = re.sub(r"\(([A-Za-z]+)\)", lambda mm: "." * len(mm.group(1)), word)
                pc = parse_coord(coord + " " + letters)
                if pc is None: continue
                r, c, d, letters = pc; dr, dc = ((0, 1), (1, 0))[d]
                if not (0 <= r < N and 0 <= c < N and r + dr * (len(letters) - 1) < N and c + dc * (len(letters) - 1) < N): continue
                main = "".join((LET[board[r + dr * j, c + dc * j]] if ch == "." else ch.upper()) for j, ch in enumerate(letters))
                widx = lex.WIDX.get(main)
                if widx is None: continue
                nums = re.findall(r"(-?\d+)\s+(-?[\d.]+)\s*$", line)
                sc, eq = (int(nums[0][0]), float(nums[0][1])) if nums else (0, 0.0)
                out.append((sc, r, c, d, len(letters), widx, letters, eq))
                continue
            if line.startswith("gid") or "game ID" in line.lower() or line.strip().startswith("Error") or (seen_header and line.strip() == ""):
                if seen_header and (line.strip() == "" or "gid" in line.lower() or "ID" in line): break
        return out
    def close(self):
        try: self.p.stdin.write("exit\n"); self.p.stdin.flush(); self.p.wait(timeout=5)
        except Exception: self.p.kill()
