"""Log-likelihoods of schemes (a) PC-FCT baseline, (b) SLT re-test to failure, (c) SLT-D.

All functions take (theta, world) and return the log-likelihood (constants in theta dropped),
exactly as derived in MODEL.md Section 3.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

import src.ig_process as ig

_GL_CACHE = {}


def _gl_nodes(n):
    if n not in _GL_CACHE:
        x, w = leggauss(n)
        b_nodes = 0.5 * (x + 1.0)  # map to (0, 1)
        b_w = 0.5 * w
        _GL_CACHE[n] = (b_nodes, b_w)
    return _GL_CACHE[n]


def _stayer_fail_block(theta, world):
    """log-lik of the stage-0 failure/censoring data common to all schemes."""
    d = world["design"]
    x0, omega, tau1, tmax = d["x0"], d["omega"], d["tau1"], d["tmax"]
    total = 0.0
    if world["d1"] > 0:
        total = total + np.sum(np.log(np.maximum(ig.passage_pdf(world["fail_pre"], x0, theta, omega), 1e-300)))
    for t in world["stayers"]:
        if t is None:
            total = total + np.log(max(1.0 - float(ig.passage_cdf(tmax, x0, theta, omega)), 1e-300))
        else:
            total = total + np.log(max(float(ig.passage_pdf(t, x0, theta, omega)), 1e-300))
    return total


def loglik_a(theta, world):
    """(a) withdrawn units discarded: each contributes S_{x0}(tau1)."""
    d = world["design"]
    x0, omega, tau1 = d["x0"], d["omega"], d["tau1"]
    s = float(ig.passage_cdf(tau1, x0, theta, omega))
    s = max(1.0 - s, 1e-300)
    return _stayer_fail_block(theta, world) + world["r1"] * np.log(s)


def continuation_cdf(u, b, x1, theta, omega):
    """P(continuation crosses omega within offset u | switch level b).

    Crossing <=> increment over (w, w+u] >= omega - b, increments are IG, so
    G(u|b) = S_IG(omega - b; mu1*dLam, lam*dLam^2), dLam = (w+u)^q - w^q.
    Vectorized over u and b (same shape)."""
    a0, a1, lam, q = theta
    u = np.asarray(u, dtype=float)
    b = np.asarray(b, dtype=float)
    w = (b / np.exp(a0 + a1 * x1)) ** (1.0 / q)
    dlam = (w + u) ** q - w**q
    out = np.where(
        dlam > 0,
        ig.ig_survival(
            np.where(dlam > 0, omega - b, 1.0),
            np.where(dlam > 0, np.exp(a0 + a1 * x1) * dlam, 1.0),
            np.where(dlam > 0, lam * dlam**2, 1.0),
        ),
        0.0,
    )
    return out


def _continuation_pdf(u, b, x1, theta, omega):
    """g(u|b) = d/du G(u|b) by central differences."""
    u = np.asarray(u, dtype=float)
    h = np.maximum(1e-7, u * 1e-5)
    lo = np.maximum(u - h, 0.0)
    hi = u + h
    return (continuation_cdf(hi, b, x1, theta, omega) - continuation_cdf(lo, b, x1, theta, omega)) / (hi - lo)


def loglik_b(theta, world, n_quad=160):
    """(b) SLT: withdrawn re-tested at x1 to failure (censored at T_max).

    Per withdrawn unit with latent level B ~ f_IG(b; a_base, b_base) on (0, omega) and
    failure offset u (or censored at u_max):
        m_f(u) = Int_0^omega f_IG(b) g(u|b) db
        m_s    = Int_0^omega f_IG(b) (1 - G(u_max|b)) db
    """
    d = world["design"]
    x0, x1, omega, tau1, tmax = d["x0"], d["x1"], d["omega"], d["tau1"], d["tmax"]
    a0, a1, lam, q = theta
    mu0 = np.exp(a0 + a1 * x0)
    a_base = mu0 * tau1**q
    b_base = lam * tau1 ** (2 * q)
    total = _stayer_fail_block(theta, world)
    if world["r1"] == 0:
        return total

    bn, bw = _gl_nodes(n_quad)  # nodes/weights on (0,1)
    b_nodes = omega * bn
    f_b = np.exp(ig.log_ig_pdf(b_nodes, a_base, b_base))  # density of latent level
    wts = bw * omega
    umax = tmax - tau1

    failed = np.array([k["z_fail"] - tau1 for k in world["withdrawn"] if k["z_fail"] is not None])
    n_cens = world["r1"] - len(failed)

    if len(failed) > 0:
        # rows: failed units, cols: quadrature nodes
        g = _continuation_pdf(failed[:, None], b_nodes[None, :], x1, theta, omega)
        m_f = np.sum(f_b[None, :] * g * wts[None, :], axis=1)
        total = total + np.sum(np.log(np.maximum(m_f, 1e-300)))
    if n_cens > 0:
        surv = 1.0 - continuation_cdf(umax, b_nodes, x1, theta, omega)
        m_s = np.sum(f_b * surv * wts)
        total = total + n_cens * np.log(max(m_s, 1e-300))
    return total


def loglik_c(theta, world):
    """(c) SLT-D: withdrawn units' baseline reading + continuation increments at x1."""
    d = world["design"]
    x0, x1, omega, tau1 = d["x0"], d["x1"], d["omega"], d["tau1"]
    a0, a1, lam, q = theta
    total = _stayer_fail_block(theta, world)
    mu0 = np.exp(a0 + a1 * x0)
    a_base = mu0 * tau1**q
    b_base = lam * tau1 ** (2 * q)
    mu1 = np.exp(a0 + a1 * x1)

    for k in world["withdrawn"]:
        b = k["B"]
        total = total + float(ig.log_ig_pdf(b, a_base, b_base))
        offs = np.asarray(k["read_offsets"], dtype=float)
        levs = np.asarray(k["read_levels"], dtype=float)
        if len(offs) == 0:
            continue
        w = (b / mu1) ** (1.0 / q)
        s = np.concatenate([[w], w + offs])
        dlam = np.diff(s**q)
        inc = np.diff(np.concatenate([[b], levs]))
        total = total + np.sum(ig.log_ig_pdf(inc, mu1 * dlam, lam * dlam**2))
    return total


def loglik(scheme, theta, world):
    if scheme == "a":
        return loglik_a(theta, world)
    if scheme == "b":
        return loglik_b(theta, world)
    if scheme == "c":
        return loglik_c(theta, world)
    if scheme == "c_nb":  # variant B: degradation follow-up without baseline reading
        from src.variant_b import loglik_c_nobaseline

        return loglik_c_nobaseline(theta, world)
    raise ValueError(scheme)
