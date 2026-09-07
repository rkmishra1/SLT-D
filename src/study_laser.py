"""Second case study: GaAs laser diodes -- heterogeneous truth, homogeneous estimation.

Real data (Meeker & Escobar 1998, p.339; GaAsLaser.csv from Meeker's repository): 15 lasers,
degradation = percent increase of operating current, readings every 250 h to 4000 h, failure
threshold omega = 10 (10% increase). The data show strong unit-to-unit rate heterogeneity
(rate CV ~ 0.23) and near-linear paths (q ~= 1), so the simulation truth is:

    mu_i ~ LogNormal(log(mu_bar) - s^2/2, s^2),  mu_bar = 0.00206, s = 0.22   (data-calibrated)
    Y_i(t) = IG(mu_i t, lam t^2)   (q = 1),  lam = 3e-5 (within-unit dispersion, data-calibrated)

Estimand: population 0.1-quantile of the failure time (marginal over mu_i), ~3600 h.
Estimator: homogeneous MLE (as a practitioner would fit), for four designs (n = 15, T_max =
4000 h, 250 h cadence -- the historical resources):
    full   : all units monitored at the use condition from t = 0   (the historical ADT)
    SLT-D  : uninstrumented life test, 50% withdrawn at tau1 = 2000 h to a rig with
             Arrhenius acceleration (Ea = 0.7 eV, 300 K -> 350 K, AF ~= 47.8), 250 h readings
    discard: pure life test (scheme (a))
The rig arm of the estimator assumes the same Arrhenius planning value (alpha1 = log AF).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm

import src.ig_process as ig
from src.fit_laser_data import OMEGA, load_laser

MU_BAR, SIG_LOG = 0.00206, 0.22
LAM, Q = 3e-5, 1.0
EA, K_B, T_USE, T_RIG = 0.7, 8.617e-5, 300.0, 350.0
AF = float(np.exp(EA / K_B * (1 / T_USE - 1 / T_RIG)))  # ~47.8
ALPHA1 = float(np.log(AF))
TMAX, TAU1, N, STEP = 4000.0, 2000.0, 15, 250.0
OFFS = np.arange(STEP, TMAX - TAU1 + 1e-9, STEP)


def true_pop_quantile(p=0.1):
    """Population p-quantile of the failure time under the heterogeneous truth (quadrature)."""
    from scipy.stats import norm

    ts = np.linspace(50.0, 20000.0, 4000)
    mu_s, w = np.polynomial.legendre.leggauss(200)
    mu_s = np.exp(np.log(MU_BAR) - SIG_LOG**2 / 2 + SIG_LOG * mu_s)
    w = w / np.sqrt(np.pi)  # Gauss-Hermite-style weights (fixed nodes trick: recompute)
    # proper lognormal quadrature: use Gauss-Hermite on z = (log mu - m)/s
    from numpy.polynomial.hermite_e import hermegauss

    z, w = hermegauss(160)
    w = w / np.sqrt(2 * np.pi)
    mus = np.exp(np.log(MU_BAR) - SIG_LOG**2 / 2 + SIG_LOG * z)
    F = np.zeros_like(ts)
    for m_i, w_i in zip(mus, w):
        a = m_i * ts
        b = LAM * ts**2
        F += w_i * ig.ig_survival(OMEGA, a, b)
    F /= np.exp(SIG_LOG**2 / 2)  # not needed with hermegauss; keep F as is
    F = np.maximum.accumulate(np.maximum(F, 0))  # enforce monotone CDF
    j = int(np.searchsorted(F, p))
    return float(np.interp(p, [F[j - 1], F[j]], [ts[j - 1], ts[j]]))


def sample_unit_passage(rng, mu_i, t_max=None):
    """First-passage time of IG(mu_i t, lam t^2) to OMEGA by bisection on the CDF."""
    th_i = np.array([np.log(mu_i), 0.0, LAM, Q])
    u = rng.random()
    if t_max is not None and u > float(ig.passage_cdf(t_max, 0.0, th_i, OMEGA)):
        return None
    lo, hi = 1e-6, 1e7
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if float(ig.passage_cdf(mid, 0.0, th_i, OMEGA)) < u:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def simulate_world_laser(rng, design):
    """design in {'full','sltd','discard'}; returns observation records per scheme."""
    mus = np.exp(np.log(MU_BAR) - SIG_LOG**2 / 2 + SIG_LOG * rng.normal(0, 1, N))
    T = [sample_unit_passage(rng, m, t_max=None) for m in mus]
    rec = {"mus": mus}
    if design == "full":
        paths = []
        for i in range(N):
            s = np.concatenate([[0.0], (np.arange(STEP, TMAX + 1e-9, STEP))**Q])
            dlam = np.diff(s)
            inc = rng.wald(mus[i] * dlam, LAM * dlam**2)
            paths.append(np.concatenate([[0.0], np.cumsum(inc)]))
        rec["paths"] = paths
        rec["times"] = np.arange(STEP, TMAX + 1e-9, STEP)
    else:
        pi1 = 0.5 if design == "sltd" else 0.0
        fail_pre = sorted(t for t in T if t is not None and t <= TAU1)
        alive = [i for i in range(N) if T[i] is None or T[i] > TAU1]
        r1 = int(np.floor(pi1 * len(alive)))
        chosen = set(rng.choice(alive, size=r1, replace=False).tolist()) if r1 else set()
        stay = [T[i] if (T[i] is not None and T[i] <= TMAX) else None
                for i in alive if i not in chosen]
        withdr = []
        for i in sorted(chosen):
            b = ig.sample_truncated_level(rng, 0.0, np.array([np.log(mus[i]), 0, LAM, Q]),
                                          OMEGA, TAU1)
            offs, levs = [], []
            cur_lev = b
            for off in OFFS:
                dl = (TAU1 + off)**Q - (TAU1 + off - STEP)**Q
                inc = rng.wald(mus[i] * AF * dl, LAM * dl**2)
                cur_lev += inc
                offs.append(off)
                levs.append(cur_lev)
                if cur_lev >= OMEGA:
                    break
            withdr.append({"B": b, "read_offsets": np.array(offs), "read_levels": np.array(levs)})
        rec.update(fail_pre=np.array(fail_pre), stay=stay, withdrawn=withdr)
    return rec


def loglik_full(theta, rec):
    mu0 = np.exp(theta[0])
    lam, q = theta[2], theta[3]
    total = 0.0
    for lev in rec["paths"]:
        s = np.concatenate([[0.0], rec["times"]**q])
        dlam = np.diff(s)
        inc = np.diff(lev)
        total += np.sum(ig.log_ig_pdf(inc, mu0 * dlam, lam * dlam**2))
    return total


def loglik_a_laser(theta, rec):
    x0, omega = 0.0, OMEGA
    total = 0.0
    for t in rec["fail_pre"]:
        total += np.log(max(float(ig.passage_pdf(t, x0, theta, omega)), 1e-300))
    for t in rec["stay"]:
        if t is None:
            total += np.log(max(1.0 - float(ig.passage_cdf(TMAX, x0, theta, omega)), 1e-300))
        else:
            total += np.log(max(float(ig.passage_pdf(t, x0, theta, omega)), 1e-300))
    return total


def loglik_c_laser(theta, rec):
    total = loglik_a_laser(theta, rec)
    a0_, a1_, lam, q = theta
    mu0 = np.exp(a0_ + a1_ * 0.0)
    a_base = mu0 * TAU1**q
    b_base = lam * TAU1**(2 * q)
    mu1 = np.exp(a0_ + a1_ * 1.0)
    for k in rec["withdrawn"]:
        b = k["B"]
        total += float(ig.log_ig_pdf(b, a_base, b_base))
        offs = np.asarray(k["read_offsets"], float)
        levs = np.asarray(k["read_levels"], float)
        if len(offs) == 0:
            continue
        w = (b / mu1)**(1.0 / q)
        s = np.concatenate([[w], w + offs])
        dlam = np.diff(s**q)
        inc = np.diff(np.concatenate([[b], levs]))
        total += np.sum(ig.log_ig_pdf(inc, mu1 * dlam, lam * dlam**2))
    return total


def fit(rec, scheme):
    if scheme == "full":
        nll = lambda z: (-(loglik_full(np.array([z[0], 0.0, np.exp(z[1]), np.exp(z[2])]), rec))
                         if np.isfinite(z).all() else 1e12)
        starts = [np.array([np.log(0.00206), np.log(3e-5), 0.0])]
    else:
        def nll(z):
            th = np.array([z[0], z[1], np.exp(z[2]), np.exp(z[3])])
            v = -(loglik_c_laser(th, rec) if scheme == "sltd" else loglik_a_laser(th, rec))
            return v if np.isfinite(v) else 1e12
        starts = [np.array([np.log(0.00206), 1.0, np.log(3e-5), 0.0]),
                  np.array([np.log(0.00206), 3.5, np.log(3e-5), 0.0])]
    best = None
    for z0 in starts:
        r = minimize(nll, z0, method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    z = best.x
    if scheme == "full":
        return np.array([z[0], 0.0, np.exp(z[1]), np.exp(z[2])])
    return np.array([z[0], z[1], np.exp(z[2]), np.exp(z[3])])


def run_cell(args):
    design, rep = args
    seed = zlib.crc32(f"{design}|{rep}".encode())
    rng = np.random.default_rng(seed)
    rec = simulate_world_laser(rng, design)
    th_hat = fit(rec, design)
    lx = float(np.log(ig.xi_p_approx(0.1, 0.0, th_hat, OMEGA)))
    return dict(design=design, rep=rep, ok=1, log_xi=lx)


if __name__ == "__main__":
    import sys

    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    # truth anchor
    xi_true = true_pop_quantile(0.1)
    print(f"AF = {AF:.1f} (Ea=0.7 eV, 300K->350K); population xi_0.1 = {xi_true:.0f} h")
    data = load_laser()
    # calibration report
    rates = np.array([ys[-1] / ts[-1] for ts, ys in data.values()])
    print(f"per-unit rates: mean={rates.mean():.5f} CV={rates.std()/rates.mean():.3f} "
          f"(log-sd={np.std(np.log(rates)):.3f})")

    cells = [(d, rep) for d in ["full", "sltd", "discard"] for rep in range(reps)]
    n_procs = max(1, os.cpu_count() - 4)
    t0 = time.time()
    out = []
    with mp.Pool(n_procs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_cell, cells, chunksize=4)):
            out.append(row)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(cells)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    df["log_xi_true"] = np.log(xi_true)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/laser_study.csv", index=False)
    g = df.groupby("design")["log_xi"]
    for d in ["full", "sltd", "discard"]:
        e = g.get_group(d) - np.log(xi_true)
        print(f"{d:8s}: bias={e.mean():+.3f} rmse={np.sqrt((e**2).mean()):.3f} "
              f"mdae={e.abs().median():.3f}")
    print("wrote results/laser_study.csv", df.shape, f"{time.time()-t0:.0f}s")
