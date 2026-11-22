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

def _tilted_G(U, P, n, phi):
    """Per-tile factors of the tilted joint P(leave) ∝ Π_x C(U_x, P_x + leave_x) · exp(phi[x, min(leave_x, 3)])."""
    G = np.zeros((27, n + 1))
    for x in range(27):
        for l in range(0, min(int(U[x]) - int(P[x]), n) + 1): G[x, l] = comb(int(U[x]), int(P[x]) + l) * np.exp(phi[x, min(l, 3)])
    return G

def _marg(G, n):
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
    return out, pre

def fit_tilt(U, P, n, target_logp, iters=30):
    """phi such that the tilted joint's per-tile marginals match exp(target_logp) (iterative proportional fitting, one tile at a
    time): of all joints over leaves of size n, the one closest to the baseline (KL) with those marginals. Returns phi and the
    largest remaining absolute marginal error."""
    T = np.exp(target_logp); phi = np.zeros((27, 4)); err = 0.0
    for _ in range(iters):
        for x in range(27):
            M, _ = _marg(_tilted_G(U, P, n, phi), n)
            ok = (M[x] > 1e-12) & (T[x] > 1e-12); phi[x, ok] += np.log(T[x, ok]) - np.log(M[x, ok])
        M, _ = _marg(_tilted_G(U, P, n, phi), n); err = np.abs(M - T).max()
        if err < 1e-4: break
    return phi, err

def sample_leaves(U, P, n, phi, k, rng):
    """k leaves (27-count vectors) drawn from the tilted joint (phi = 0: the exact baseline) by backward sampling over the DP."""
    G = _tilted_G(U, P, n, phi); _, pre = _marg(G, n); out = np.zeros((k, 27), dtype=np.int64)
    for s in range(k):
        r = n
        for x in range(26, -1, -1):
            w = np.array([G[x, l] * pre[x][r - l] if l <= r else 0.0 for l in range(n + 1)])
            l = int(rng.choice(n + 1, p=w / w.sum())); out[s, x] = l; r -= l
    return out
