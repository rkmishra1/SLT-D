"""k-stage study: does spreading the withdrawal over several stage-change times help?

Designs share (n=20, x0=0.46, omega=10, tmax=5000, read_step=50) and are matched in expected
number of withdrawn units (~4.6-5.0):
  K=1: tau=(2000),        pi=(0.50), x=(0.78)
  K=2: tau=(1500,3000),   pi=(0.30,0.30), x=(0.78,1.00)
  K=3: tau=(1000,2000,3500), pi=(0.15,0.20,0.25), x=(0.78,0.90,1.00)
Estimand: log xi_0.1(0). 500 reps.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import src.ig_process as ig
import src.kstage as ks
import src.mle as mle

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
DESIGNS = {
    "K1": [(2000.0, 0.50, 0.78)],
    "K2": [(1500.0, 0.30, 0.78), (3000.0, 0.30, 1.00)],
    "K3": [(1000.0, 0.15, 0.78), (2000.0, 0.20, 0.90), (3500.0, 0.25, 1.00)],
}


def build(name):
    stages = DESIGNS[name]
    return {
        "x0": 0.46, "omega": 10.0, "n": 20, "tmax": 5000.0, "stages": stages,
        "read_offsets": np.arange(50.0, 5000.0 - stages[-1][0] + 1.0, 50.0),
    }


def run_cell(args):
    name, rep = args
    d = build(name)
    seed = zlib.crc32(f"{name}|{rep}".encode())
    rng = np.random.default_rng(seed)
    w = ks.simulate_world_kstage(rng, THETA, d)
    log_xi_true = float(np.log(ig.xi_p_approx(0.1, 0.0, THETA, 10.0)))

    def nll(z):
        th = np.array([z[0], z[1], np.exp(z[2]), np.exp(z[3])])
        v = -ks.loglik_c_kstage(th, w)
        return v if np.isfinite(v) else 1e12

    starts = mle.make_starts(tuple(THETA), n_starts=4, seed=rep)
    best = None
    for z0 in starts:
        r = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    th = np.array([best.x[0], best.x[1], np.exp(best.x[2]), np.exp(best.x[3])])
    n_w = sum(len(c["units"]) for c in w["cohorts"])
    n_r = sum(len(u["read_levels"]) for c in w["cohorts"] for u in c["units"])
    return dict(design=name, rep=rep, ok=1, log_xi=float(np.log(ig.xi_p_approx(0.1, 0.0, th, 10.0))),
                log_xi_true=log_xi_true, n_withdrawn=n_w, n_readings=n_r,
                n_cross=sum(1 for c in w["cohorts"] for u in c["units"]
                            if len(u["read_levels"]) and u["read_levels"][-1] >= 10.0))


def main(reps=500):
    cells = [(name, rep) for name in DESIGNS for rep in range(reps)]
    n_procs = max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_cell, cells, chunksize=4)):
            out.append(row)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(cells)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/kstage_study.csv", index=False)
    print("wrote results/kstage_study.csv", df.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
