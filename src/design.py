"""Design theory for SLT-D: expected Fisher information and V-optimal plans.

Expected information per test unit (scheme c, all-L-readings planning approximation):

  I_unit = Int_0^tau1  f0(t) Hf(t) dt                                   pre-tau1 failures
         + (1-pi1) [ Int_tau1^Tmax f0(t) Hf(t) dt + S0(Tmax) HS ]       stayers
         + pi1 Int_0^omega fB(b) [ Hbase(b) + Sum_j J_j(b)' W_j(b) J_j(b) ] db   withdrawn

with Hf(t) = -d^2 log f0(t), HS = -d^2 log S0(Tmax), Hbase(b) = -d^2 log f_IG(b; a0(tau1), b0(tau1)),
and, for continuation increment j with mean a_j and shape s_j (both functions of theta and the
observed baseline b), the expected information J' W J, where W = diag(s/a^3, 1/(2 s^2)) is the
IG(a, s) Fisher information and J = d(a_j, s_j)/d(theta) (the second-derivative term vanishes
in expectation because the IG score has mean zero).

Avar(log xi_p) = g' (n I_unit)^{-1} g,  g = d log xi_p / d theta.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

import src.ig_process as ig

_H_CACHE = {}


def _gl(n, lo, hi):
    key = (n, lo, hi)
    if key not in _H_CACHE:
        x, w = leggauss(n)
        _H_CACHE[key] = (lo + (hi - lo) * (x + 1) / 2, (hi - lo) * w / 2)
    return _H_CACHE[key]


def hessian_fd(func, theta, h):
    """4x4 central-difference Hessian of a scalar function at theta (vectorized-friendly)."""
    theta = np.asarray(theta, float)
    n = len(theta)
    f0 = func(theta)

    def sf(z):
        v = func(z)
        return v if np.isfinite(v) else 1e12

    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (sf(theta + ei) - 2 * f0 + sf(theta - ei)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (sf(theta + ei + ej) - sf(theta + ei - ej)
                                 - sf(theta - ei + ej) + sf(theta - ei - ej)) / (4 * h[i] * h[j])
    return -H  # observed information


def vec_neg_hessian(logf_vec, theta, h, nodes):
    """Per-node negative central-difference Hessian of a vectorized log-density.

    logf_vec(z, nodes) -> array over nodes. Returns array of shape (4, 4, n_nodes).
    """
    theta = np.asarray(theta, float)
    n = len(theta)

    def sf(z):
        v = logf_vec(z, nodes)
        return np.where(np.isfinite(v), v, 1e12)

    f0 = sf(theta)
    H = np.zeros((n, n, len(nodes)))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (sf(theta + ei) - 2 * f0 + sf(theta - ei)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (sf(theta + ei + ej) - sf(theta + ei - ej)
                                 - sf(theta - ei + ej) + sf(theta - ei - ej)) / (4 * h[i] * h[j])
    return -H


def expected_info_unit(theta, design, n_t=48, n_b=96, include_baseline=True):
    """Expected Fisher information (4x4) per unit for scheme (c)."""
    theta = np.asarray(theta, float)
    a0_, a1_, lam, q = theta
    x0, x1, omega = design["x0"], design["x1"], design["omega"]
    tau1, tmax, pi1 = design["tau1"], design["tmax"], design["pi1"]
    # note: log f0 is itself a numerical t-derivative, so the theta-FD steps must stay well
    # above the nested-differentiation noise floor (eps ~ 1e-12): h^2 >> 1e-12.
    h = np.array([1e-3, 1e-3, 1e-3, 1e-4])

    I = np.zeros((4, 4))

    # ---- stage-0 failure/censoring block (common to all schemes)
    def logf0_vec(z, t):
        return np.log(np.maximum(ig.passage_pdf(t, x0, z, omega), 1e-300))

    for lo, hi in [(1e-6, tau1), (tau1, tmax)]:
        t, wt = _gl(n_t, lo, hi)
        Hn = vec_neg_hessian(logf0_vec, theta, h, t)  # (4,4,n_t)
        dens = ig.passage_pdf(t, x0, theta, omega)
        I += np.einsum("ijn,jn->ij", Hn, dens * wt[None, :]) * (1.0 if lo == 1e-6 else (1.0 - pi1))

    # stayer censored at T_max: point mass S0(T_max) * HS
    def logS(z):
        return np.log(max(1.0 - float(ig.passage_cdf(tmax, x0, z, omega)), 1e-300))

    HS = hessian_fd(logS, theta, h)
    I += (1.0 - pi1) * float(1.0 - ig.passage_cdf(tmax, x0, theta, omega)) * HS

    # ---- withdrawn block
    b, wb = _gl(n_b, 0.0, omega)
    fB = np.exp(ig.log_ig_pdf(b, np.exp(a0_ + a1_ * x0) * tau1**q, lam * tau1 ** (2 * q)))

    if include_baseline:
        def logbase_vec(z, bb):
            return ig.log_ig_pdf(bb, np.exp(z[0] + z[1] * x0) * tau1 ** z[3], z[2] * tau1 ** (2 * z[3]))

        Hb = vec_neg_hessian(logbase_vec, theta, h, b)
        I += pi1 * np.einsum("ijn,jn->ij", Hb, fB * wb[None, :])

    # continuation increments: expected info via J' W J per increment per baseline node
    offs = np.concatenate([[0.0], np.asarray(design["read_offsets"], float)])
    mu1 = np.exp(a0_ + a1_ * x1)

    def inc_params(z, bb):
        w = (bb / np.exp(z[0] + z[1] * x1)) ** (1.0 / z[3])
        s = w[None, :] + offs[:, None]  # (L+1, n_b)
        dlam = np.diff(s ** z[3], axis=0)  # (L, n_b)
        a = np.exp(z[0] + z[1] * x1) * dlam
        sh = z[2] * dlam**2
        return a, sh  # (L, n_b)

    a_nodes, s_nodes = inc_params(theta, b)
    L = a_nodes.shape[0]
    J = np.zeros((4, 2, L, len(b)))  # J[k, m, l, n] = d param_m / d theta_k, increment l, node n
    for k in range(4):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += h[k]
        tm[k] -= h[k]
        ap, sp = inc_params(tp, b)
        am, sm = inc_params(tm, b)
        J[k, 0] = (ap - am) / (2 * h[k])
        J[k, 1] = (sp - sm) / (2 * h[k])

    W = np.zeros((2, 2, L, len(b)))
    W[0, 0] = s_nodes / a_nodes**3
    W[1, 1] = 1.0 / (2.0 * s_nodes**2)

    # reading l (interval (u_{l-1}, u_l]) is observed iff the unit has not crossed omega by
    # u_{l-1}: weight by 1 - G(u_{l-1} | b) (continuation crossing CDF); the crossing reading
    # itself is still an ordinary IG observation, so no further term is needed.
    from src.likelihood import continuation_cdf

    alive = np.ones((L, len(b)))
    for l in range(1, L):
        alive[l] = 1.0 - continuation_cdf(offs[l - 1], b, x1, theta, omega)

    # I_inc[i, j] = sum_l int fB(b) alive_l(b) (J_l' W_l J_l)_{ij} db
    I_inc = np.einsum("imln,mpln,jpln,ln,n->ij", J, W, J, alive, fB * wb)
    I += pi1 * I_inc
    return I


def _psd_clip(M, floor_rel=1e-8):
    """Clip tiny negative eigenvalues (numerical) to a small positive floor."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    floor = floor_rel * max(float(w[-1]), 1e-12)
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def avar_log_xi(theta, design, n, p=0.1, x_use=0.0, info=None):
    theta = np.asarray(theta, float)
    if info is None:
        info = expected_info_unit(theta, design)
    tot = _psd_clip(n * info)
    g = _grad_log_xi(p, x_use, theta, design["omega"])
    try:
        cov = np.linalg.inv(tot)
    except np.linalg.LinAlgError:
        return np.inf
    v = float(g @ cov @ g)
    return v if v > 0 else np.inf


def _grad_log_xi(p, x, theta, omega):
    from src.mle import grad_log_xi_p

    return grad_log_xi_p(p, x, theta, omega)


# ------------------------------------------------------------------ ADT comparator

def info_unit_adt(theta, x, times, omega):
    """Expected info per unit for a fresh constant-stress ADT unit measured at `times` (from 0)."""
    theta = np.asarray(theta, float)
    lam, q = theta[2], theta[3]
    mu = np.exp(theta[0] + theta[1] * x)
    h = np.array([1e-4, 1e-4, 1e-6, 1e-6])
    tt = np.concatenate([[0.0], np.asarray(times, float)])
    dlam0 = np.diff(tt**q)

    def params(z):
        d = np.diff(tt ** z[3])
        return np.exp(z[0] + z[1] * x) * d, z[2] * d**2

    a0n, s0n = params(theta)
    L = len(a0n)
    J = np.zeros((4, 2, L))
    for k in range(4):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += h[k]
        tm[k] -= h[k]
        ap, sp = params(tp)
        am, sm = params(tm)
        J[k, 0] = (ap - am) / (2 * h[k])
        J[k, 1] = (sp - sm) / (2 * h[k])
    W = np.zeros((2, 2, L))
    W[0, 0] = s0n / a0n**3
    W[1, 1] = 1.0 / (2.0 * s0n**2)
    return np.einsum("iml,mpl,jpl->ij", J, W, J)
