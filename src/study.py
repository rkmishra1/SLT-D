"""Pilot simulation study: information hierarchy of schemes (a)/(b)/(c) across scenarios.

Scenarios (see MODEL.md Section 5), true theta from the real-data fit, reps share one latent
world across schemes (common random numbers).
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import src.ig_process as ig
import src.mle as mle
from src.simulate import simulate_world

THETA = (-1.8944, 1.7377, 0.6285, 0.4491)  # our real-data fit

SCENARIOS = {
    "S1-connector": dict(x0=0.78, x1=1.0, omega=30.0, n=20, tau1=2000.0, tmax=5000.0,
                         read_step=150.0),
    "S2-moderate": dict(x0=0.46, x1=0.78, omega=10.0, n=20, tau1=2000.0, tmax=5000.0,
                        read_step=50.0),
    "S3-severe": dict(x0=0.46, x1=0.78, omega=30.0, n=30, tau1=2000.0, tmax=5000.0,
                      read_step=150.0),
}


def build_design(scen, pi1):
    s = SCENARIOS[scen]
    return {
        "x0": s["x0"], "x1": s["x1"], "omega": s["omega"], "n": s["n"],
        "tau1": s["tau1"], "tmax": s["tmax"], "pi1": pi1,
        "read_offsets": np.arange(s["read_step"], s["tmax"] - s["tau1"] + 1.0, s["read_step"]),
    }


def run_cell(args):
    import zlib

    scen, pi1, rep = args
    design = build_design(scen, pi1)
    # deterministic across processes/runs (hash() of str is randomized per interpreter)
    seed = zlib.crc32(f"{scen}|{pi1}|{rep}".encode())
    rng = np.random.default_rng(seed)
    world = simulate_world(rng, np.array(THETA), design)
    log_xi_true = float(np.log(ig.xi_p_approx(0.1, 0.0, np.array(THETA), design["omega"])))
    rows = []
    for scheme in ["a", "b", "c"]:
        starts = mle.make_starts(THETA, n_starts=4, seed=rep)
        fit = mle.fit_mle(world, scheme, starts)
        if fit is None:
            rows.append(dict(scenario=scen, pi1=pi1, rep=rep, scheme=scheme, d1=world["d1"],
                             r1=world["r1"], ok=0))
            continue
        inf = mle.xi_inference(fit, scheme, world, 0.1, 0.0, design["omega"])
        n_with_fail_b = sum(1 for k in world["withdrawn"] if k["z_fail"] is not None)
        n_cross_c = sum(1 for k in world["withdrawn"]
                        if len(k["read_levels"]) and k["read_levels"][-1] >= design["omega"])
        rows.append(dict(
            scenario=scen, pi1=pi1, rep=rep, scheme=scheme, d1=world["d1"], r1=world["r1"],
            ok=1, n_with_fail_b=n_with_fail_b, n_cross_c=n_cross_c,
            a0=fit["theta"][0], a1=fit["theta"][1], lam=fit["theta"][2], q=fit["theta"][3],
            log_xi=inf["log_xi"], se=inf["se"], lo=inf["lo"], hi=inf["hi"],
            log_xi_true=log_xi_true,
        ))
    return rows


def main(reps=200, n_procs=None, out_csv="results/per_rep.csv"):
    cells = [(sc, pi, rep)
             for sc in ["S1-connector", "S3-severe"]
             for pi in [0.5]
             for rep in range(reps)]
    cells += [(sc, pi, rep)
              for sc in ["S2-moderate"]
              for pi in [0.25, 0.5, 0.75]
              for rep in range(reps)]
    n_procs = n_procs or max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_cell, cells, chunksize=4)):
            out.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(cells)} cells done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", df.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "results/per_rep.csv"
    main(reps, out_csv=out_csv)
