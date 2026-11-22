"""Rack-inference helpers (round 6) without torch: the no-inference baseline."""
import numpy as np
from math import comb

def baseline_logp(U, P, n):
    """log P(kept count of tile x = 0 / 1 / 2 / 3+) with P(leave) ∝ Π_x C(U_x, P_x + leave_x), |leave| = n."""
    G = np.zeros((27, n + 1))
    for x in range(27):
        for l in range(0, min(int(U[x]) - int(P[x]), n) + 1): G[x, l] = comb(int(U[x]), int(P[x]) + l)
    one = np.zeros(n + 1); one[0] = 1.0; mul = lambda a, b: np.convolve(a, b)[:n + 1]
    pre = [one]
    for x in range(27): pre.append(mul(pre[-1], G[x]))
    suf = [one] * 28
    for x in range(26, -1, -1): suf[x] = mul(G[x], suf[x + 1])
    Z = pre[27][n]; out = np.zeros((27, 4))
    for x in range(27):
        rest = mul(pre[x], suf[x + 1])
        for l in range(n + 1):
            if G[x, l] > 0: out[x, min(l, 3)] += G[x, l] * rest[n - l] / Z
    return np.log(np.clip(out, 1e-9, 1.0))
