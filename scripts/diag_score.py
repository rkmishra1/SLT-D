"""Diagnose the score-identity failure in schemes (b) and (c).

Decomposes the score into blocks (stage-0 failure block, baseline, continuation) and uses
paired common-random-number comparisons of E[ell(theta)] vs E[ell(theta +/- h)].
"""
import numpy as np

import src.ig_process as ig
import src.likelihood as lik
from src.simulate import simulate_world

theta = (-1.88, 1.73, 0.653, 0.449)
design = {
    "x0": 0.78, "x1": 1.0, "omega": 30.0, "n": 12, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
    "read_offsets": np.arange(150.0, 3001.0, 150.0),
}

rng = np.random.default_rng(123)
reps = 600
worlds = [simulate_world(rng, theta, design) for _ in range(reps)]

h = np.array([1e-4, 1e-4, 1e-6, 1e-6]) * np.array([1.0, 1.0, 0.01, 0.01])


def score_of(scheme, w):
    s = np.zeros(4)
    for i in range(4):
        tp = np.array(theta, float); tm = np.array(theta, float)
        tp[i] += h[i]; tm[i] -= h[i]
        s[i] = (lik.loglik(scheme, tp, w) - lik.loglik(scheme, tm, w)) / (2 * h[i])
    return s


for scheme in ["a", "b", "c"]:
    S = np.array([score_of(scheme, w) for w in worlds])
    print(f"--- scheme {scheme} (reps={reps}) ---")
    print("mean   :", np.round(S.mean(axis=0), 4))
    print("median :", np.round(np.median(S, axis=0), 4))
    print("se(mean):", np.round(S.std(axis=0) / np.sqrt(reps), 4))
    print("mean d1:", np.mean([w['d1'] for w in worlds]), " mean r1:", np.mean([w['r1'] for w in worlds]))

# block decomposition for scheme (c)
from src.likelihood import _stayer_fail_block

bl, ba, bc = [], [], []
for w in worlds:
    d = w["design"]
    x0, x1, omega, tau1 = d["x0"], d["x1"], d["omega"], d["tau1"]
    a0, a1, lam, q = theta
    mu0 = np.exp(a0 + a1 * x0); mu1 = np.exp(a0 + a1 * x1)
    tot_fail = tot_base = tot_cont = np.zeros(4)
    def grad(f):
        g = np.zeros(4)
        for i in range(4):
            tp = np.array(theta, float); tm = np.array(theta, float)
            tp[i] += h[i]; tm[i] -= h[i]
            g[i] = (f(tp) - f(tm)) / (2 * h[i])
        return g
    for k in w["withdrawn"]:
        b = k["B"]
        offs = np.asarray(k["read_offsets"], float); levs = np.asarray(k["read_levels"], float)
        wb = (b / mu1) ** (1 / q)
        def f_base(th):
            return float(ig.log_ig_pdf(b, np.exp(th[0] + th[1] * x0) * tau1 ** th[3], th[2] * tau1 ** (2 * th[3])))
        def f_cont(th):
            ww = (b / np.exp(th[0] + th[1] * x1)) ** (1 / th[3])
            s_ = np.concatenate([[ww], ww + offs])
            dl = np.diff(s_ ** th[3])
            inc = np.diff(np.concatenate([[b], levs]))
            return float(np.sum(ig.log_ig_pdf(inc, np.exp(th[0] + th[1] * x1) * dl, th[2] * dl ** 2)))
        tot_base += grad(f_base)
        tot_cont += grad(f_cont)
    tot_fail = grad(lambda th: _stayer_fail_block(th, w))
    bl.append(tot_fail); ba.append(tot_base); bc.append(tot_cont)

bl, ba, bc = np.array(bl), np.array(ba), np.array(bc)
print("\n=== scheme (c) block decomposition ===")
print("fail-block  mean:", np.round(bl.mean(0), 4), "se:", np.round(bl.std(0)/np.sqrt(reps), 4))
print("baseline    mean:", np.round(ba.mean(0), 4), "se:", np.round(ba.std(0)/np.sqrt(reps), 4))
print("cont-increm mean:", np.round(bc.mean(0), 4), "se:", np.round(bc.std(0)/np.sqrt(reps), 4))
