"""Bayesian planning under parameter uncertainty.

Prior: from the pilot study (the real connector data, 18 units) -- a multivariate normal on the
unconstrained z = (alpha0, alpha1, log lam, log q) centered at the pilot MLE with covariance
inv(H_pilot), the inverse observed information of the pilot fit (optionally scaled by `kappa`
to represent a pilot of different size).

Criterion (Bayesian V-optimality / pre-posterior variance averaging, cf. Li et al. 2017):
    J(D) = E_{theta ~ prior} [ Avar( log xi_hat_p(0); theta, D ) ],
with Avar from the planning information (Section 5 of the paper) at fixed design D (n fixed by
the budget at the planning values; theta only enters the criterion).

Outputs: plug-in plan (base theta) vs Bayesian-optimal plan, each with prior-expected Avar,
90th percentile of Avar over the prior draws, and Avar at the true theta.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import src.ig_process as ig
from src.design import _grad_log_xi, _psd_clip, avar_log_xi, expected_info_unit
from src.fit_real_data import nll as pilot_nll

THETA_TRUE = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
THETA_BASE = THETA_TRUE.copy()  # planning values (here: the pilot MLE)
OMEGA, TMAX, P, BUDGET, C_B, C_MEA, N_MAX = 30.0, 5000.0, 0.1, 6000.0, 200.0, 5.0, 30


# ------------------------------------------------------------------ prior

def pilot_prior(kappa=1.0):
    """Prior N(z_hat, kappa * inv(H_pilot)) on z-space from the real-data pilot fit."""
    z_hat = np.array([THETA_BASE[0], THETA_BASE[1], np.log(THETA_BASE[2]), np.log(THETA_BASE[3])])
    h = np.array([1e-3, 1e-3, 1e-3, 1e-4])
    H = np.zeros((4, 4))
    f0 = pilot_nll(z_hat)
    for i in range(4):
        ei = np.zeros(4)
        ei[i] = h[i]
        H[i, i] = (pilot_nll(z_hat + ei) - 2 * f0 + pilot_nll(z_hat - ei)) / h[i] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            ei = np.zeros(4)
            ej = np.zeros(4)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (pilot_nll(z_hat + ei + ej) - pilot_nll(z_hat + ei - ej)
                                 - pilot_nll(z_hat - ei + ej) + pilot_nll(z_hat - ei - ej)) / (4 * h[i] * h[j])
    cov = np.linalg.inv(H)
    cov = 0.5 * (cov + cov.T)
    w = np.linalg.eigvalsh(cov)
    if w[0] <= 0:  # jitter tiny negative directions
        cov += (abs(w[0]) + 1e-8) * np.eye(4)
    return z_hat, kappa * cov, np.sqrt(np.diag(cov))


def prior_draws(z_hat, cov, m, seed=7):
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(z_hat, cov, size=m)
    return np.array([[z[0], z[1], np.exp(z[2]), np.exp(z[3])] for z in Z])


# ------------------------------------------------------------------ criterion

def build_design(x0, tau1, pi1, x1, step, n=1):
    return {"x0": x0, "x1": x1, "omega": OMEGA, "n": n, "tau1": tau1, "tmax": TMAX, "pi1": pi1,
            "read_offsets": np.arange(step, TMAX - tau1 + 1e-9, step)}


def expected_readings_per_unit(d, theta):
    s0 = float(1.0 - ig.passage_cdf(d["tau1"], d["x0"], theta, OMEGA))
    return d["pi1"] * s0 * len(d["read_offsets"])


def feasible_n(d, theta_plan):
    per_unit = C_B + C_MEA * expected_readings_per_unit(d, theta_plan)
    return min(int(np.floor(BUDGET / per_unit)), N_MAX)


def bayes_criterion(d, thetas, g_cache=None):
    """Mean over prior draws of Avar(log xi_p(0); theta, d)."""
    vals = []
    for th in thetas:
        g = _grad_log_xi(P, 0.0, th, OMEGA)
        I = expected_info_unit(th, d)
        try:
            cov = np.linalg.inv(_psd_clip(d["n"] * I))
            v = float(g @ cov @ g)
        except np.linalg.LinAlgError:
            v = np.inf
        vals.append(v if np.isfinite(v) else 1e12)
    return np.array(vals)


def grid(x0s=None, tau1s=None, pi1s=None, x1s=None, steps=None):
    return itertools.product(
        x0s if x0s is not None else [0.30, 0.46, 0.62, 0.78],
        tau1s if tau1s is not None else np.arange(500.0, 4200.0, 250.0),
        pi1s if pi1s is not None else np.arange(0.1, 0.95, 0.1),
        x1s if x1s is not None else [0.85, 1.0],
        steps if steps is not None else [150.0, 500.0],
    )


def _eval_combo(args):
    (x0, tau1, pi1, x1, step), thetas, z_hat = args
    d = build_design(x0, tau1, pi1, x1, step)
    if x1 <= x0 + 0.05:
        return None
    n = feasible_n(d, THETA_BASE)
    if n < 4:
        return None
    d["n"] = n
    vals = bayes_criterion(d, thetas)
    return (x0, tau1, pi1, x1, step, n, float(vals.mean()), float(np.quantile(vals, 0.9)),
            float(avar_log_xi(THETA_TRUE, d, n, p=P, x_use=0.0)))


def optimize(thetas, z_hat, grd, n_procs=None):
    tasks = [(combo, thetas, z_hat) for combo in grd]
    n_procs = n_procs or max(1, os.cpu_count() - 4)
    out = []
    with mp.Pool(n_procs) as pool:
        for r in pool.imap_unordered(_eval_combo, tasks, chunksize=4):
            if r is not None:
                out.append(r)
    return out


def summarize_best(rows, key=lambda r: r[6]):
    return min(rows, key=key)


if __name__ == "__main__":
    t0 = time.time()
    result = {"scenarios": {}}
    for kappa in [1.0, 4.0]:
        z_hat, cov, sds = pilot_prior(kappa=kappa)
        print(f"\n=== kappa={kappa} (prior sd on z: {np.round(sds, 3)}) ===")
        thetas = prior_draws(z_hat, cov, m=400, seed=7)
        grd = list(grid(x0s=[0.10, 0.15, 0.20, 0.30, 0.46, 0.62, 0.78], tau1s=np.arange(500.0, 4200.0, 250.0),
                        pi1s=np.arange(0.1, 0.95, 0.1), x1s=[0.85, 1.0], steps=[150.0, 500.0]))
        rows = optimize(thetas, z_hat, grd)
        print(f"evaluated {len(rows)} feasible combos ({time.time() - t0:.0f}s)")

        bayes = summarize_best(rows, key=lambda r: r[6])      # min prior-expected Avar
        plugin = summarize_best(rows, key=lambda r: r[8])     # min Avar at prior mean

        d_b = build_design(*bayes[:5])
        d_b["n"] = bayes[5]
        d_p = build_design(*plugin[:5])
        d_p["n"] = plugin[5]
        vals_b = bayes_criterion(d_b, thetas)
        vals_p = bayes_criterion(d_p, thetas)

        result["scenarios"][f"kappa{int(kappa)}"] = {
            "bayes_plan": [float(x) for x in bayes],
            "plugin_plan": [float(x) for x in plugin],
            "bayes_eval": {"J_mean": float(vals_b.mean()), "J_p90": float(np.quantile(vals_b, 0.9)),
                            "avar_true": float(avar_log_xi(THETA_TRUE, d_b, bayes[5], p=P, x_use=0.0))},
            "plugin_eval": {"J_mean": float(vals_p.mean()), "J_p90": float(np.quantile(vals_p, 0.9)),
                             "avar_true": float(avar_log_xi(THETA_TRUE, d_p, plugin[5], p=P, x_use=0.0))},
        }
        print("bayes-optimal:", np.round(bayes[:6], 3), "J=", round(bayes[6], 4),
              "p90=", round(result['scenarios'][f'kappa{int(kappa)}']['bayes_eval']['J_p90'], 4))
        print("plug-in:      ", np.round(plugin[:6], 3), "J=", round(vals_p.mean(), 4),
              "p90=", round(result['scenarios'][f'kappa{int(kappa)}']['plugin_eval']['J_p90'], 4))

    os.makedirs("results", exist_ok=True)
    with open("results/bayes_design.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote results/bayes_design.json", f"({time.time() - t0:.0f}s)")
