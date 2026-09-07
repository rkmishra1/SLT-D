"""Validate design.py: (1) IG(a,s) Fisher info; (2) expected per-unit info vs the average
observed information of simulated scheme-(c) worlds; (3) Asvar(log xi) vs empirical variance."""
import numpy as np

import src.ig_process as ig
import src.likelihood as lik
import src.mle as mle
from src.design import expected_info_unit, avar_log_xi, info_unit_adt
from src.simulate import simulate_world

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])


def check_ig_info():
    """E[-d^2 log f / d(a,s)^2] = diag(s/a^3, 1/(2 s^2)) (cross = 0) by quadrature over y."""
    a, s = 2.0, 5.0
    ys = np.linspace(1e-4, a + 12 * np.sqrt(a**3 / s), 400_001)
    f = np.exp(ig.log_ig_pdf(ys, a, s))
    # numeric second derivatives of log f at a grid? cheaper: verify expected info identity via
    # E[score^2] = -E[d^2 log f] using FD derivatives at quadrature nodes.
    h = 1e-4
    d2_aa = (ig.log_ig_pdf(ys, a + h, s) - 2 * ig.log_ig_pdf(ys, a, s)
             + ig.log_ig_pdf(ys, a - h, s)) / h**2
    d2_ss = (ig.log_ig_pdf(ys, a, s + h) - 2 * ig.log_ig_pdf(ys, a, s) + ig.log_ig_pdf(ys, a, s - h)) / h**2
    w = f / np.trapezoid(f, ys)
    I_aa = -np.trapezoid(d2_aa * w, ys)
    I_ss = -np.trapezoid(d2_ss * w, ys)
    print(f"IG info: I_aa numeric={I_aa:.4f} analytic={s/a**3:.4f}; "
          f"I_ss numeric={I_ss:.4f} analytic={1/(2*s**2):.4f}")
    assert abs(I_aa / (s / a**3) - 1) < 0.01 and abs(I_ss / (1 / (2 * s**2)) - 1) < 0.01


def check_expected_info(reps=60):
    design = {
        "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 12, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
        "read_offsets": np.arange(50.0, 3001.0, 50.0),
    }
    rng = np.random.default_rng(99)
    Hs = []
    for _ in range(reps):
        w = simulate_world(rng, THETA, design)
        # observed info at true theta via FD Hessian of loglik in z-space mapped to theta-space:
        # simpler: average the Hessian in the unconstrained z then transform. To avoid transform
        # subtleties, compute Hessian directly in theta-space with likelihood in theta.
        h = np.array([1e-4, 1e-4, 1e-6, 1e-6])

        def f(th):
            v = lik.loglik("c", th, w)
            return v if np.isfinite(v) else 1e12

        th0 = THETA.copy()
        H = np.zeros((4, 4))
        for i in range(4):
            ei = np.zeros(4)
            ei[i] = h[i]
            H[i, i] = -(f(th0 + ei) - 2 * f(th0) + f(th0 - ei)) / h[i] ** 2
        for i in range(4):
            for j in range(i + 1, 4):
                ei = np.zeros(4)
                ej = np.zeros(4)
                ei[i] = h[i]
                ej[j] = h[j]
                H[i, j] = H[j, i] = -(f(th0 + ei + ej) - f(th0 + ei - ej) - f(th0 - ei + ej)
                                      + f(th0 - ei - ej)) / (4 * h[i] * h[j])
        Hs.append(H)
    emp = np.mean(Hs, axis=0) / design["n"]  # per-unit
    ana = expected_info_unit(THETA, design)
    print("\nper-unit expected info (analytic):")
    print(np.round(ana, 4))
    print("per-unit observed info (empirical):")
    print(np.round(emp, 4))
    rel = np.abs(emp - ana) / (np.abs(ana) + 1e-9)
    print("relative |diff| (offdiag can be noisy):", np.round(rel, 3))
    # eigenvalue comparison (what matters for Avar)
    ev_a = np.linalg.eigvalsh(ana)[::-1]
    ev_e = np.linalg.eigvalsh(emp)[::-1]
    print("eigenvalues analytic:", np.round(ev_a, 4))
    print("eigenvalues empirical:", np.round(ev_e, 4))
    assert np.allclose(ev_a, ev_e, rtol=0.15), "eigenvalue mismatch"


def check_avar_vs_empirical():
    design = {
        "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 20, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
        "read_offsets": np.arange(50.0, 3001.0, 50.0),
    }
    import pandas as pd

    df = pd.read_csv("results/per_rep_1000.csv")
    g = df[(df.scenario == "S2-moderate") & (df.pi1 == 0.5) & (df.scheme == "c") & (df.ok == 1)]
    emp_var = float(np.var(g["log_xi"], ddof=1))
    pred = avar_log_xi(THETA, design, design["n"])
    print(f"\nS2 pi=0.5 scheme c: empirical var(log xi_hat)={emp_var:.4f}  predicted Asvar={pred:.4f}")


if __name__ == "__main__":
    check_ig_info()
    check_expected_info()
    try:
        check_avar_vs_empirical()
    except FileNotFoundError:
        print("\n(1000-rep study not finished yet; skipping Asvar check)")
