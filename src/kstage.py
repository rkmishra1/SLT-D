"""k-stage SLT-D: multiple withdrawal times, each with its own degradation rig.

Design: stages = [(tau_1, pi_1, x_1), ..., (tau_K, pi_K, x_K)] with
tau_1 < ... < tau_K < tmax and x_j > x0. At tau_j, a floor(pi_j * #survivors-on-test)
randomized subset of the survivors is withdrawn to a rig at stress x_j, where a baseline
reading and periodic readings (offsets shared by all cohorts) are taken until crossing or
the horizon. Remaining units stay on stage 0 until failure or tmax.

Likelihood (ignorable withdrawal at every stage; Lemma 1 of the paper persists):
    ell = sum_{failures} log f_0(t)  +  n_cens * log S_0(tmax)
        + sum_j sum_{k in cohort j} [ log f_IG(b_k; a_0(tau_j), b_0(tau_j))
                                      + sum_l log f_IG(Delta_kl; mu_{x_j} dLam, lam dLam^2) ]
The baseline density f_IG(b) restricted to b < omega already carries the stage-0 survival
term S_0(tau_j) (mass of the level density below omega equals S_0(tau_j)), so no separate
survival term is added for withdrawn units -- exactly as in the 2-stage scheme (c).
"""
from __future__ import annotations

import numpy as np

import src.ig_process as ig


def simulate_world_kstage(rng, theta, design_k):
    x0, omega, n, tmax = design_k["x0"], design_k["omega"], design_k["n"], design_k["tmax"]
    stages = design_k["stages"]  # [(tau_j, pi_j, x_j), ...] ascending tau
    offsets = np.asarray(design_k["read_offsets"], float)

    a0, a1, lam, q = theta
    mu0 = np.exp(a0 + a1 * x0)

    # exact stage-0 lifetimes
    T = np.array([ig.sample_passage_time(rng, x0, theta, omega) for _ in range(n)])

    on_test = np.array([True] * n)  # still on the life test
    failures = []
    cohorts = []
    for (tau_j, pi_j, x_j) in stages:
        # failures in (prev tau, tau_j]
        newly_failed = on_test & np.array([t is not None and t <= tau_j for t in T])
        failures += [float(T[i]) for i in np.where(newly_failed)[0]]
        on_test &= ~newly_failed
        # withdraw
        survivors = np.where(on_test)[0]
        r_j = int(np.floor(pi_j * len(survivors)))
        if r_j > 0:
            chosen = rng.choice(survivors, size=r_j, replace=False)
            on_test[chosen] = False
            mu_j = np.exp(a0 + a1 * x_j)
            units = []
            for i in sorted(chosen):
                b = ig.sample_truncated_level(rng, x0, theta, omega, tau_j)
                w = float(ig.virtual_age(b, x_j, theta))
                levels = []
                cur = w
                lev = b
                for off in offsets:
                    nxt = w + off
                    dlam = nxt**q - cur**q
                    inc = rng.wald(mu_j * dlam, lam * dlam**2)
                    lev = lev + inc
                    cur = nxt
                    levels.append(lev)
                    if lev >= omega:
                        break
                units.append({"B": b, "read_offsets": offsets[: len(levels)],
                              "read_levels": np.array(levels)})
            cohorts.append({"tau": tau_j, "x": x_j, "units": units})
    # remaining failures up to tmax and censoring
    final_failed = on_test & np.array([t is not None and t <= tmax for t in T])
    failures += [float(T[i]) for i in np.where(final_failed)[0]]
    on_test &= ~final_failed
    n_cens = int(on_test.sum())

    return {"design": design_k, "failures": np.sort(np.array(failures)),
            "n_cens": n_cens, "cohorts": cohorts}


def loglik_c_kstage(theta, world):
    x0, omega, tmax = world["design"]["x0"], world["design"]["omega"], world["design"]["tmax"]
    a0, a1, lam, q = theta
    mu0 = np.exp(a0 + a1 * x0)

    total = 0.0
    if len(world["failures"]):
        total += np.sum(np.log(np.maximum(ig.passage_pdf(world["failures"], x0, theta, omega), 1e-300)))
    if world["n_cens"] > 0:
        total += world["n_cens"] * np.log(max(1.0 - float(ig.passage_cdf(tmax, x0, theta, omega)), 1e-300))

    for coh in world["cohorts"]:
        tau_j, x_j = coh["tau"], coh["x"]
        mu_j = np.exp(a0 + a1 * x_j)
        a_base = mu0 * tau_j**q
        b_base = lam * tau_j ** (2 * q)
        for k in coh["units"]:
            b = k["B"]
            total += float(ig.log_ig_pdf(b, a_base, b_base))
            offs = np.asarray(k["read_offsets"], float)
            levs = np.asarray(k["read_levels"], float)
            if len(offs) == 0:
                continue
            w = (b / mu_j) ** (1.0 / q)
            s = np.concatenate([[w], w + offs])
            dlam = np.diff(s**q)
            inc = np.diff(np.concatenate([[b], levs]))
            total += np.sum(ig.log_ig_pdf(inc, mu_j * dlam, lam * dlam**2))
    return total
