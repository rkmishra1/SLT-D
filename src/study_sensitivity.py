"""Estimation-robustness: does the hierarchy (c) > (b) > (a) survive true-parameter
misspecification? L9(3^4) array of +-10% errors on (alpha0, alpha1, lambda, q) applied to the
TRUE data-generating theta (planning/fitting uses the base theta); S2 scenario, 300 reps.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import src.ig_process as ig
import src.mle as mle
from src.simulate import simulate_world

BASE = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
DESIGN = {
    "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 20, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
    "read_offsets": np.arange(50.0, 3001.0, 50.0),
}
L9 = [
    (1, 1, 1, 1), (1, 2, 2, 2), (1, 3, 3, 3),
    (2, 1, 2, 3), (2, 2, 3, 1), (2, 3, 1, 2),
    (3, 1, 3, 2), (3, 2, 1, 3), (3, 3, 2, 1),
]


def run_cell(args):
    combo_idx, rep = args
    combo = L9[combo_idx]
    eps = np.array([{1: -0.10, 2: 0.0, 3: 0.10}[c] for c in combo])
    theta_true = BASE * (1.0 + eps)
    rng = np.random.default_rng(hash((combo_idx, rep)) % (2**32))
    world = simulate_world(rng, theta_true, DESIGN)
    log_xi_true = float(np.log(ig.xi_p_approx(0.1, 0.0, theta_true, 10.0)))
    rows = []
    for scheme in ["b", "c"]:
        starts = mle.make_starts(tuple(BASE), n_starts=6, seed=rep, scale=0.30)
        fit = mle.fit_mle(world, scheme, starts)
        if fit is None:
            rows.append(dict(combo=combo_idx, rep=rep, scheme=scheme, ok=0))
            continue
        inf = mle.xi_inference(fit, scheme, world, 0.1, 0.0, 10.0)
        rows.append(dict(combo=combo_idx, rep=rep, scheme=scheme, ok=1,
                         log_xi=inf["log_xi"], log_xi_true=log_xi_true,
                         a0=fit["theta"][0], a1=fit["theta"][1], lam=fit["theta"][2], q=fit["theta"][3]))
    return rows


def main(reps=300):
    cells = [(ci, rep) for ci in range(9) for rep in range(reps)]
    n_procs = max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_cell, cells, chunksize=4)):
            out.extend(rows)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(cells)} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/sensitivity_estimation.csv", index=False)
    print("wrote results/sensitivity_estimation.csv", df.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
