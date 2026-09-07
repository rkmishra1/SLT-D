"""Recompute scheme-(c) SEs from the expected information at theta-hat (always PD, consistent
with the planning theory); scheme-(b) rows keep the observed-information SEs already stored.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import src.ig_process as ig
import src.mle as mle
from src.design import expected_info_unit, _psd_clip
from src.study import SCENARIOS, THETA


def build_design(scen, pi1):
    s = SCENARIOS[scen]
    return {
        "x0": s["x0"], "x1": s["x1"], "omega": s["omega"], "n": s["n"],
        "tau1": s["tau1"], "tmax": s["tmax"], "pi1": pi1,
        "read_offsets": np.arange(s["read_step"], s["tmax"] - s["tau1"] + 1.0, s["read_step"]),
    }


def run_cell(args):
    scen, pi1, rep, theta_hat, log_xi = args
    design = build_design(scen, pi1)
    I = expected_info_unit(np.array(theta_hat), design)
    tot = _psd_clip(design["n"] * I)
    g = mle.grad_log_xi_p(0.1, 0.0, np.array(theta_hat), design["omega"])
    try:
        var = float(g @ np.linalg.inv(tot) @ g)
    except np.linalg.LinAlgError:
        var = np.nan
    from scipy.stats import norm

    if not np.isfinite(var) or var <= 0:
        return dict(scenario=scen, pi1=pi1, rep=rep, se=np.nan, lo=np.nan, hi=np.nan, finite=0)
    se = float(np.sqrt(var))
    cr = float(norm.ppf(0.975))
    return dict(scenario=scen, pi1=pi1, rep=rep, se=se, lo=log_xi - cr * se,
                hi=log_xi + cr * se, finite=1)


def main():
    df = pd.read_csv("results/per_rep_1000_fixed.csv")
    sub = df[(df.ok == 1) & (df.scheme == "c")].copy()
    cells = [(r.scenario, float(r.pi1), int(r.rep), (r.a0, r.a1, r.lam, r.q), float(r.log_xi))
             for r in sub.itertuples()]
    n_procs = max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_cell, cells, chunksize=8)):
            out.append(row)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(cells)} ({time.time() - t0:.0f}s)", flush=True)
    se = pd.DataFrame(out)
    df = df.merge(se, on=["scenario", "pi1", "rep"], how="left", suffixes=("", "_exp"))
    is_c = df.scheme == "c"
    df.loc[is_c, "se"] = df.loc[is_c, "se_exp"]
    df.loc[is_c, "lo"] = df.loc[is_c, "lo_exp"]
    df.loc[is_c, "hi"] = df.loc[is_c, "hi_exp"]
    df.loc[is_c, "se_finite"] = df.loc[is_c, "finite"]
    df = df.drop(columns=[c for c in ["se_exp", "lo_exp", "hi_exp", "finite"] if c in df.columns])
    df.to_csv("results/per_rep_1000_fixed.csv", index=False)
    print(f"wrote results/per_rep_1000_fixed.csv with expected-info SEs for scheme c ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
