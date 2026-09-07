"""Variant B: scheme (c) without the baseline reading at withdrawal.

The withdrawn unit's baseline level B is latent; the observed-data likelihood per unit is
    L_k = Int_0^omega f_IG(b; a0(tau1), b0(tau1)) * prod_j f_IG(Delta_kj | b) db,
with continuation increments as in scheme (c) (path-level CE, virtual age w(b; theta)).

EM view (MODEL.md remark): complete data = (b, increments); the E-step conditional density of
b given the increments is proportional to the integrand above (no closed form -> quadrature),
and the M-step maximizes the expected complete log-likelihood numerically (theta enters the
increment means through w(b; theta), so no closed-form M-step). Direct maximization of the
quadrature-approximated marginal likelihood is the implemented, equivalent estimator.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

import src.ig_process as ig
from src.likelihood import _gl_nodes, _stayer_fail_block


def loglik_c_nobaseline(theta, world, n_quad=120):
    """Scheme-(c) likelihood with the baseline reading integrated out (variant B)."""
    d = world["design"]
    x0, x1, omega, tau1 = d["x0"], d["x1"], d["omega"], d["tau1"]
    a0, a1, lam, q = theta
    mu0 = np.exp(a0 + a1 * x0)
    mu1 = np.exp(a0 + a1 * x1)
    a_base = mu0 * tau1**q
    b_base = lam * tau1 ** (2 * q)
    total = _stayer_fail_block(theta, world)
    if world["r1"] == 0:
        return total

    nodes01, wts01 = _gl_nodes(n_quad)
    b_nodes = omega * nodes01  # latent baseline grid
    log_prior = ig.log_ig_pdf(b_nodes, a_base, b_base)  # density of B (untruncated; support handled below)
    logw = np.log(wts01 * omega)

    for k in world["withdrawn"]:
        offs = np.asarray(k["read_offsets"], dtype=float)
        levs = np.asarray(k["read_levels"], dtype=float)
        if len(offs) == 0:
            total += logsumexp(log_prior + logw)
            continue
        # continuation transformed-time boundaries at each node: [w, w+u_1, ..., w+u_L]
        w = (b_nodes / mu1) ** (1.0 / q)
        s = w[:, None] + np.concatenate([[0.0], offs])[None, :]
        dlam = np.diff(s**q, axis=1)  # (nodes, L)
        # increment matrix: first column is lev1 - b (latent baseline, node-dependent);
        # later columns are the observed reading-to-reading increments.
        inc_mat = np.empty_like(dlam)
        inc_mat[:, 0] = levs[0] - b_nodes
        inc_mat[:, 1:] = np.diff(levs)[None, :]
        log_lik_inc = ig.log_ig_pdf(inc_mat, mu1 * dlam, lam * dlam**2).sum(axis=1)
        total += logsumexp(log_prior + log_lik_inc + logw)

    return total
