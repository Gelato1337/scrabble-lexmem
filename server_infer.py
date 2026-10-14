"""GPU inference server for the Go self-play engine (cmd/lexmem_selfplay in our Macondo checkout).
Protocol (little-endian, one request -> one response, in order, on a Unix socket):
  request : u32 npos, u32 nmoves_total, then
            boards u8[npos*225] (letter idx + 1, 0 = empty), racks u8[npos*27], unseen u8[npos*27], glob f32[npos*3]
            ([bag/100, (my score - opp score)/100, turn/30]), nmoves u16[npos],
            widx i32[T], mf f32[T*9] (columns 0-8 of the move features; column 8 = exchange flag), leave u8[T*27]
  response: f32 logits[T] (policy head), f32 values[npos*2] (TD value head: spread change / 100, win logit)
Also writes data/lexwords.txt (our lexicon's word order, which widx indexes) at start.
Usage: CKPT=data/policy_X.pt SOCK=/tmp/lexmem.sock python server_infer.py"""
import os, socket, struct, time, numpy as np, torch
from lexmem.core import Lexicon
from lexmem.policy import PolicyNet, MF
torch.backends.mha.set_fastpath_enabled(False)
E = os.environ.get; dev = "cuda" if torch.cuda.is_available() else "cpu"
lex = Lexicon(E("LEXICON", "data/NWL23.txt"))
open("data/lexwords.txt", "w").write("\n".join(lex.words) + "\n")
model = PolicyNet(lex).to(dev); model.load_state_dict(torch.load(E("CKPT"), map_location=dev), strict=False); model.eval()
print(f"model {E('CKPT')} on {dev}; {lex.NW} words", flush=True)
def recv_exact(conn, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0: raise ConnectionError("client closed")
        got += k
    return buf
@torch.no_grad()
def infer(npos, T, raw):
    o = 0
    def take(dtype, count):
        nonlocal o; a = np.frombuffer(raw, dtype=dtype, count=count, offset=o); o += a.nbytes; return a
    boards = take(np.uint8, npos * 225).reshape(npos, 15, 15); racks = take(np.uint8, npos * 27).reshape(npos, 27)
    unseen = take(np.uint8, npos * 27).reshape(npos, 27); glob = take(np.float32, npos * 3).reshape(npos, 3)
    nm = take(np.uint16, npos).astype(np.int64); widx = take(np.int32, T); mf9 = take(np.float32, T * 9).reshape(T, 9); leave = take(np.uint8, T * 27).reshape(T, 27)
    M = int(nm.max()); starts = np.concatenate([[0], np.cumsum(nm)[:-1]])
    pos = np.concatenate([np.arange(n) for n in nm]); row = np.repeat(np.arange(npos), nm)
    W = np.zeros((npos, M), dtype=np.int64); F = np.zeros((npos, M, MF), dtype=np.float32); Lv = np.zeros((npos, M, 27), dtype=np.float32); V = np.zeros((npos, M), dtype=bool)
    W[row, pos] = widx; F[row, pos, :9] = mf9; Lv[row, pos] = leave; V[row, pos] = True
    b = dict(board=torch.from_numpy(boards.astype(np.int64)), rack=torch.from_numpy(racks.astype(np.float32)), unseen=torch.from_numpy(unseen.astype(np.float32)),
             glob=torch.from_numpy(glob.copy()), mf=torch.from_numpy(F), leave=torch.from_numpy(Lv), widx=torch.from_numpy(W), valid=torch.from_numpy(V))
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        pl, q, _ = model.heads(b)
        _, v = model.qv({k: b[k] for k in ("board", "rack", "unseen", "glob")}, with_moves=False)   # value as trained: position-only pass
    pl = pl.float().cpu().numpy(); v = v.float().cpu().numpy()
    return pl[row, pos].astype(np.float32).tobytes() + v.astype(np.float32).tobytes()
sock_path = E("SOCK", "/tmp/lexmem.sock")
if os.path.exists(sock_path): os.remove(sock_path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(sock_path); srv.listen(8)
print(f"listening on {sock_path}", flush=True)
while True:
    conn, _ = srv.accept(); n_req = n_pos = 0; t0 = time.time()
    try:
        while True:
            npos, T = struct.unpack("<II", recv_exact(conn, 8))
            size = npos * (225 + 27 + 27 + 12 + 2) + T * (4 + 36 + 27)
            conn.sendall(infer(npos, T, recv_exact(conn, size))); n_req += 1; n_pos += npos
            if n_req % 500 == 0: print(f"{n_req} batches, {n_pos} positions, {n_pos/(time.time()-t0):.0f} positions/s", flush=True)
    except ConnectionError:
        print(f"client closed after {n_req} batches / {n_pos} positions", flush=True); conn.close()
