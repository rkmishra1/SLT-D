"""Optimal SLT-D design under the sequential constraint (stage-0 is a failure-only life test),
compared with the unconstrained two-level parallel ADT at the same budget.

Design variables: x0 (stage-0 stress), tau1, pi1, x1 (rig stress), reading step, n.
Objective: Asvar(log xi_0.1 at use condition) from the validated planning information.
Benchmarks and outputs:
  (1) constrained-optimal SLT-D plan;
  (2) unconstrained two-level parallel ADT plan (readings at two constant stresses);
  (3) L9 (+-10%) robustness of the SLT-D plan to planning-value errors.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import src.ig_process as ig
from src.design import avar_log_xi, info_unit_adt, _grad_log_xi, _psd_clip

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
OMEGA = 30.0
TMAX = 5000.0
P = 0.1


def build_design(x0, tau1, pi1, x1, step):
    return {
        "x0": x0, "x1": x1, "omega": OMEGA, "n": 1, "tau1": tau1, "tmax": TMAX, "pi1": pi1,
        "read_offsets": np.arange(step, TMAX - tau1 + 1e-9, step),
    }


def expected_readings_per_unit(design, theta):
    s0 = float(1.0 - ig.passage_cdf(design["tau1"], design["x0"], theta, OMEGA))
    return design["pi1"] * s0 * len(design["read_offsets"])


def grid(x0s=None, tau1s=None, pi1s=None, x1s=None, steps=None):
    return itertools.product(
        x0s if x0s is not None else [0.10, 0.15, 0.20, 0.30, 0.46, 0.62, 0.78],
        tau1s if tau1s is not None else np.arange(500.0, 4200.0, 250.0),
        pi1s if pi1s is not None else np.arange(0.1, 0.95, 0.1),
        x1s if x1s is not None else [0.85, 0.925, 1.0],
        steps if steps is not None else [150.0, 500.0],
    )


def optimize(c_b=200.0, c_mea=5.0, budget=6000.0, n_max=30, theta=THETA, grd=None, verbose=True):
    best = None
    for x0, tau1, pi1, x1, step in (grd or grid()):
        if x1 <= x0 + 0.05:
            continue
        d = build_design(x0, tau1, pi1, x1, step)
        per_unit = c_b + c_mea * expected_readings_per_unit(d, theta)
        n = min(int(np.floor(budget / per_unit)), n_max)
        if n < 4:
            continue
        v = avar_log_xi(theta, d, n, p=P, x_use=0.0)
        if best is None or v < best[-1]:
            best = (x0, tau1, pi1, x1, step, n, v)
    if verbose and best:
        print(f"optimal SLT-D: x0={best[0]:.2f} tau1={best[1]:.0f} pi1={best[2]:.2f} x1={best[3]:.2f} "
              f"step={best[4]:.0f} n={best[5]} -> Asvar={best[6]:.4f} (std e^{np.sqrt(best[6]):.2f})")
    return best


def adt_plan(c_b=200.0, c_mea=5.0, budget=6000.0, n_max=30, theta=THETA,
             xs=None, splits=None, step=150.0):
    xs = xs if xs is not None else np.arange(0.0, 1.01, 0.1)
    splits = splits if splits is not None else np.arange(0.25, 0.8, 0.125)
    g = _grad_log_xi(P, 0.0, theta, OMEGA)
    times = np.arange(step, TMAX + 1e-9, step)
    L = len(times)
    per_unit = c_b + c_mea * L
    n = min(int(np.floor(budget / per_unit)), n_max)
    best = None
    for xa, xb, s in itertools.product(xs, xs, splits):
        if xb <= xa + 0.1:
            continue
        n1 = max(1, int(round(s * n)))
        n2 = n - n1
        if n2 < 1:
            continue
        I = n1 * info_unit_adt(theta, xa, times, OMEGA) + n2 * info_unit_adt(theta, xb, times, OMEGA)
        v = float(g @ np.linalg.inv(_psd_clip(I)) @ g)
        if best is None or v < best[-1]:
            best = (xa, xb, step, n1, n2, v)
    return best


def l9(theta_plan_values, c_b=200.0, c_mea=5.0, budget=6000.0, n_max=30):
    l9arr = [
        (1, 1, 1, 1), (1, 2, 2, 2), (1, 3, 3, 3),
        (2, 1, 2, 3), (2, 2, 3, 1), (2, 3, 1, 2),
        (3, 1, 3, 2), (3, 2, 1, 3), (3, 3, 2, 1),
    ]
    grd = list(grid(x0s=[0.10, 0.20, 0.46, 0.78], tau1s=np.arange(500.0, 4200.0, 250.0),
                    pi1s=np.arange(0.1, 0.95, 0.1), x1s=[0.85, 1.0], steps=[150.0, 500.0]))
    out = []
    for combo in l9arr:
        eps = np.array([{1: -0.10, 2: 0.0, 3: 0.10}[c] for c in combo])
        th_plan = theta_plan_values * (1.0 + eps)
        best = optimize(c_b, c_mea, budget, n_max, theta=th_plan, grd=grd, verbose=False)
        d = build_design(*best[:5])
        v_true = avar_log_xi(THETA, d, best[5], p=P, x_use=0.0)
        out.append(dict(eps=[float(e) for e in eps], plan=[float(x) for x in best[:6]],
                        asvar_at_true=float(v_true)))
        print(f"eps={np.round(eps, 2)} plan(x0,tau1,pi1,x1,step,n)={np.round(best[:6], 2)} "
              f"Asvar@true={v_true:.4f}")
    return out


if __name__ == "__main__":
    print("=== constrained-optimal SLT-D plan ===")
    base = optimize()
    print("\n=== unconstrained two-level parallel ADT, same budget ===")
    adt = adt_plan()
    print(f"optimal ADT: xa={adt[0]:.2f} xb={adt[1]:.2f} n=({adt[3]},{adt[4]}) step={adt[2]:.0f} "
          f"-> Asvar={adt[5]:.4f} (std e^{np.sqrt(adt[5]):.2f})")
    print(f"price of the sequential constraint (ADT/SLT-D Asvar ratio): {adt[5] / base[6]:.2f}")
    print("\n=== L9 (+-10%) plan robustness ===")
    l9out = l9(THETA)
    with open("results/design_results.json", "w") as f:
        json.dump(dict(base=[float(x) for x in base], adt=[float(x) for x in adt], l9=l9out), f, indent=2)
    print("wrote results/design_results.json")
