"""Polite crawler for cross-tables self-annotated games. 1 request/second.
  index: all index pages -> data/xt/index.jsonl  (id, lexicon, players, bingos)
  gcg  : download GCGs for ids in the index matching LEXICONS -> data/xt/gcg/<id>.gcg (skips existing)"""
import re, sys, time, json, os, urllib.request
UA = {"User-Agent": "scrabble-lexmem research crawler (contact: jmlaat@utu.fi)"}
def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode("utf-8", "ignore")
def index():
    out = open("data/xt/index.jsonl", "a"); seen = set()
    if os.path.exists("data/xt/index.jsonl"):
        for l in open("data/xt/index.jsonl"): seen.add(json.loads(l)["id"])
    for off in range(1, 51200, 100):
        try: h = get(f"https://www.cross-tables.com/annolistself.php?offset={off}")
        except Exception as e: print("err", off, e, flush=True); time.sleep(5); continue
        rows = h.split("annotated.php?u=")[1:]
        for r in rows:
            gid = int(re.match(r"(\d+)", r).group(1))
            if gid in seen: continue
            t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", r[:3000]))
            lx = re.search(r"\b(NWL\d+|TWL\d+|THTWL\d+|CSW\d+|[A-Z]{2,5}\d{2})\b", t)
            pl = re.search(r"\|\s*([^|]+?)\s*\|\s*vs\.\s*\|\s*([^|]+?)\s*\|", t)
            seen.add(gid); out.write(json.dumps(dict(id=gid, lexicon=lx.group(1) if lx else None,
                                     p1=pl.group(1) if pl else None, p2=pl.group(2) if pl else None)) + "\n")
        out.flush(); print(f"offset {off}: total {len(seen)}", flush=True); time.sleep(1)
def gcgs(lexicons):
    os.makedirs("data/xt/gcg", exist_ok=True)
    ids = [json.loads(l) for l in open("data/xt/index.jsonl")]
    ids = [x["id"] for x in ids if x["lexicon"] and any(x["lexicon"].startswith(p) for p in lexicons)]
    print(len(ids), "games to fetch", flush=True); n = 0
    for gid in ids:
        p = f"data/xt/gcg/{gid}.gcg"
        if os.path.exists(p): continue
        try: open(p, "w").write(get(f"https://www.cross-tables.com/annotated/selfgcg/{gid // 100}/anno{gid}.gcg"))
        except Exception as e: print("err", gid, e, flush=True)
        n += 1; time.sleep(1)
        if n % 500 == 0: print(f"{n} fetched", flush=True)
if __name__ == "__main__":
    if sys.argv[1] == "index": index()
    else: gcgs(sys.argv[2].split(","))
