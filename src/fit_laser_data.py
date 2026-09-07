"""Second case study: GaAs laser diodes (Meeker & Escobar 1998, p. 339).

15 lasers, degradation = percent increase in operating current, readings every 250 h
(0, 250, ..., 4000 h); failure = 10% increase (omega = 10). Single condition (no stress
link): fit (mu, lambda, q) via the IG increment likelihood; the use-condition quantile is
computed at x = 0 with alpha1 = 0 for reporting (alpha1 is a planning quantity in the design
comparison of src/study_laser.py).
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.optimize import minimize

import src.ig_process as ig

OMEGA = 10.0


def load_laser(path="data_GaAsLaser.csv"):
    units = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            u = int(row["Unit Number"])
            units.setdefault(u, ([], []))
            units[u][0].append(float(row["Hours"]))
            units[u][1].append(float(row["Percent Increase"]))
    return {u: (np.asarray(ts, float), np.asarray(ys, float)) for u, (ts, ys) in units.items()}


def nll_laser(z, data):
    """Negative log-likelihood of IG increments at the (single) use condition.
    z = (log mu, log lam, log q). Y(0)=0, so all readings give ordinary increments."""
    mu, lam, q = np.exp(z)
    total = 0.0
    for ts, ys in data.values():
        s = ts**q
        dlam = np.diff(s)
        inc = np.diff(ys)
        total -= np.sum(ig.log_ig_pdf(inc, mu * dlam, lam * dlam**2))
    return total if np.isfinite(total) else 1e12


def fit_laser(path="data_GaAsLaser.csv"):
    data = load_laser(path)
    best = None
    rng = np.random.default_rng(0)
    for k in range(16):
        z0 = np.array([np.log(0.05), np.log(50.0), np.log(1.0)]) if k == 0 else \
            np.array([rng.uniform(-4, 0), rng.uniform(1, 5), rng.uniform(-0.7, 0.7)])
        r = minimize(nll_laser, z0, args=(data,), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        r = minimize(nll_laser, r.x, args=(data,), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-11})
        if best is None or r.fun < best.fun:
            best = r
    mu, lam, q = np.exp(best.x)
    theta = np.array([np.log(mu), 0.0, lam, q])  # alpha1 = 0 (single condition)
    xi = float(ig.xi_p_approx(0.1, 0.0, theta, OMEGA))
    xi_exact = float(ig.passage_quantile(0.1, 0.0, theta, OMEGA)[0])
    # observed crossings for context
    n_cross = sum(int((ys >= OMEGA).any()) for _, ys in data.values())
    return dict(mu=mu, lam=lam, q=q, nll=float(best.fun), theta=theta,
                xi01_approx=xi, xi01_exact=xi_exact, n_units=len(data), n_cross=n_cross,
                data=data)


if __name__ == "__main__":
    fit = fit_laser()
    print(f"units: {fit['n_units']}   crossings observed by 4000 h: {fit['n_cross']}")
    print(f"fit: mu={fit['mu']:.5f}  lambda={fit['lam']:.3f}  q={fit['q']:.4f}  (nll {fit['nll']:.3f})")
    print(f"xi_0.1 (10% current increase): approx={fit['xi01_approx']:.0f} h, exact={fit['xi01_exact']:.0f} h")
    # median and MTTF for context
    th = fit["theta"]
    print("median:", round(float(ig.xi_p_approx(0.5, 0.0, th, OMEGA)), 0),
          " MTTF~E[T]:", round(float(ig.passage_quantile(0.5, 0.0, th, OMEGA)[0]), 0))
    # rough check: fraction failing by 4000 h under the fit vs observed 3/15
    print("F(4000) under fit:", round(float(ig.passage_cdf(4000.0, 0.0, th, OMEGA)), 3), "(observed 3/15)")
