"""Read a wolges/Macondo KWG file and enumerate the DAWG words (node 0 = DAWG root)."""
import numpy as np, sys
def read_kwg(path):
    a = np.fromfile(path, dtype="<u4")
    arc = a & 0x3FFFFF; is_end = (a >> 22) & 1; accepts = (a >> 23) & 1; tile = a >> 24
    return arc, is_end, accepts, tile
def words(path, alphabet="?ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    arc, is_end, accepts, tile = read_kwg(path)
    out = []
    stack = [(int(arc[0]), "")]          # children of DAWG root start at arc[0]
    while stack:
        p, prefix = stack.pop()
        if p == 0: continue
        while True:
            w = prefix + alphabet[tile[p]]
            if accepts[p]: out.append(w)
            if arc[p]: stack.append((int(arc[p]), w))
            if is_end[p]: break
            p += 1
    return out
if __name__ == "__main__":
    ws = words(sys.argv[1]); ws.sort()
    open(sys.argv[2], "w").write("\n".join(ws) + "\n")
    print(len(ws), "words; sample:", ws[:3], ws[len(ws)//2], ws[-3:])
