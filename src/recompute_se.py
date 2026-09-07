"""Recompute Wald SEs for the 1000-rep study with the fixed observed-information routine
(FD steps above the nested-derivative noise floor; z->theta covariance transform).

Point estimates (theta-hat) are reused from per_rep_1000.csv; worlds are regenerated
deterministically from the same seeds as study.py.
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
from src.study import SCENARIOS, THETA


def build_design(scen, pi1):
    s = SCENARIOS[scen]
    return {
        "x0": s["x0"], "x1": s["x1"], "omega": s["omega"], "n": s["n"],
        "tau1": s["tau1"], "tmax": s["tmax"], "pi1": pi1,
        "read_offsets": np.arange(s["read_step"], s["tmax"] - s["tau1"] + 1.0, s["read_step"]),
    }


def run_cell(args):
    scen, pi1, rep, scheme, theta_hat, log_xi = args
    design = build_design(scen, pi1)
    rng = np.random.default_rng(hash((scen, pi1, rep)) % (2**32))
    world = simulate_world(rng, np.array(THETA), design)
    z = np.array([theta_hat[0], theta_hat[1], np.log(theta_hat[2]), np.log(theta_hat[3])])
    cov_z = mle.observed_info_cov(z, scheme, world)
    if cov_z is None:
        return dict(scenario=scen, pi1=pi1, rep=rep, scheme=scheme,
                    se=np.nan, lo=np.nan, hi=np.nan, finite=0)
    cov_t = mle.cov_theta_from_z(cov_z, np.array(theta_hat))
    g = mle.grad_log_xi_p(0.1, 0.0, np.array(theta_hat), design["omega"])
    var = float(g @ cov_t @ g)
    from scipy.stats import norm

    if not np.isfinite(var) or var <= 0:
        return dict(scenario=scen, pi1=pi1, rep=rep, scheme=scheme,
                    se=np.nan, lo=np.nan, hi=np.nan, finite=0)
    se = float(np.sqrt(var))
    cr = float(norm.ppf(0.975))
    return dict(scenario=scen, pi1=pi1, rep=rep, scheme=scheme,
                se=se, lo=log_xi - cr * se, hi=log_xi + cr * se, finite=1)


def main():
    df = pd.read_csv("results/per_rep_1000.csv")
    df = df[(df.ok == 1) & (df.scheme.isin(["b", "c"]))].copy()
    cells = [(r.scenario, float(r.pi1), int(r.rep), r.scheme,
              (r.a0, r.a1, r.lam, r.q), float(r.log_xi))
             for r in df.itertuples()]
    n_procs = max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_cell, cells, chunksize=16)):
            out.append(row)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(cells)} ({time.time() - t0:.0f}s)", flush=True)
    se = pd.DataFrame(out)
    base = pd.read_csv("results/per_rep_1000.csv")
    base = base.merge(se, on=["scenario", "pi1", "rep", "scheme"], how="left",
                      suffixes=("", "_new"))
    base["se"] = base["se_new"]
    base["lo"] = base["lo_new"]
    base["hi"] = base["hi_new"]
    base["se_finite"] = base["finite"]
    base = base.drop(columns=[c for c in ["se_new", "lo_new", "hi_new", "finite"] if c in base.columns])
    base.to_csv("results/per_rep_1000_fixed.csv", index=False)
    print(f"wrote results/per_rep_1000_fixed.csv ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
