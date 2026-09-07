"""Simulation of one latent SLT-D world; schemes (a)/(b)/(c) are observation masks over it.

World layout (all quantities under true theta):
  n units at stress x0; lifetimes T_i (exact, by inversion).
  D1 = failures at/before tau1 -> fail_pre (sorted times).
  At tau1: r1 = floor(pi1 * (n - d1)) survivors withdrawn at random (fixed draw, shared by schemes).
  Withdrawn unit k: baseline level B_k (level at tau1 under x0, truncated < omega);
    - scheme (b) continuation failure time at x1 (exact, by inversion from virtual age);
    - scheme (c) readings at offsets u_1..u_L (exact increments), stopped at first reading >= omega.
  Stayers: lifetime T_i, censored at T_max.
"""
from __future__ import annotations

import numpy as np

import src.ig_process as ig


def simulate_world(rng, theta, design):
    x0 = design["x0"]
    x1 = design["x1"]
    omega = design["omega"]
    tau1 = design["tau1"]
    tmax = design["tmax"]
    n = design["n"]
    pi1 = design["pi1"]
    read_offsets = design["read_offsets"]  # strictly increasing offsets after tau1

    # stage-0 lifetimes (shared by all schemes)
    T = np.array([ig.sample_passage_time(rng, x0, theta, omega) for _ in range(n)])

    alive_mask = (T > tau1) | np.array([t is None for t in T])
    d1 = int(np.sum(T <= tau1))
    fail_pre = np.sort(T[T <= tau1])

    survivors = np.where(alive_mask)[0]
    r1 = int(np.floor(pi1 * len(survivors)))
    withdrawn_idx = rng.choice(survivors, size=r1, replace=False)
    withdrawn_set = set(withdrawn_idx.tolist())
    stayers = [i for i in range(n) if (i not in withdrawn_set) and alive_mask[i]]

    withdrawn = []
    umax = tmax - tau1
    for k in sorted(withdrawn_idx):
        b = ig.sample_truncated_level(rng, x0, theta, omega, tau1)
        w = float(ig.virtual_age(b, x1, theta))
        # scheme (b): continuation failure by inversion of G(u|b) = P(cross within offset u | b)
        from src.likelihood import continuation_cdf

        z_fail = None
        uu = rng.random()
        g_max = float(continuation_cdf(umax, b, x1, theta, omega))
        if uu <= g_max:
            lo, hi = 1e-9, umax
            for _ in range(120):
                mid = 0.5 * (lo + hi)
                if float(continuation_cdf(mid, b, x1, theta, omega)) < uu:
                    lo = mid
                else:
                    hi = mid
            z_fail = tau1 + 0.5 * (lo + hi)
        # scheme (c): readings by exact increments; stop at first reading >= omega
        levels = []
        cur = w
        lev = b
        for off in read_offsets:
            nxt = w + off
            dlam = nxt**theta[3] - cur**theta[3]
            inc = rng.wald(ig.mu_of(x1, theta) * dlam, theta[2] * dlam**2)
            lev = lev + inc
            cur = nxt
            levels.append(lev)
            if lev >= omega:
                break

        withdrawn.append({"B": b, "z_fail": z_fail, "read_offsets": read_offsets[: len(levels)], "read_levels": np.array(levels)})

    stay = []
    for i in stayers:
        t = T[i]
        stay.append(None if (t is None or t > tmax) else float(t))  # None = alive at T_max

    return {
        "design": design,
        "d1": d1,
        "r1": r1,
        "fail_pre": fail_pre,
        "stayers": stay,
        "withdrawn": withdrawn,
    }
