"""Core IG-process mathematics for the SLT-D model.

Parameterization (Ma et al. 2021 / Wang et al. 2016):
    theta = (alpha0, alpha1, lam, q)
    mu_x  = exp(alpha0 + alpha1 * x)              # stress link, standardized stress x in [0,1]
    Lam(t) = t**q
    Y_x(t) ~ IG(a, b) with a = mu_x * Lam(t), b = lam * Lam(t)**2
    increments: IG(mu_x * dLam, lam * dLam**2), dLam = t2**q - t1**q

IG(a, b) density: f(y) = sqrt(b / (2 pi y^3)) exp(-b (y - a)^2 / (2 a^2 y)),  E = a, Var = a^3 / b.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr

NEG_INF = -np.inf


def unpack(theta):
    return float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])


def mu_of(x: float, theta) -> float:
    a0, a1, _, _ = unpack(theta)
    return float(np.exp(a0 + a1 * x))


def log_ig_pdf(y, a, b):
    """log density of IG(a, b) evaluated at y > 0 (vectorized; no clipping of inputs)."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bad = ~((y > 0) & np.isfinite(y) & (a > 0) & (b > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * np.log(b) - 0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(y) - b * (y - a) ** 2 / (2.0 * a * a * y)
    out = np.where(bad, NEG_INF, out)
    return out


def ig_survival(y, a, b):
    """P(IG(a,b) > y), numerically stable; y, a, b > 0, vectorized."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.sqrt(b / y)
    u1 = r * (1.0 - y / a)
    u2 = -(r * (1.0 + y / a))
    # second term: exp(2b/a) * Phi(u2) = exp(2b/a + log Phi(u2)); the combined exponent
    # behaves like -b (y-a)^2 / (2 a^2 y) <= 0, so exp is safe.
    z = 2.0 * b / a + log_ndtr(u2)
    z = np.minimum(z, 0.0)  # guard against tiny positive values from rounding
    term2 = np.exp(z)
    s = ndtr(u1) - term2
    return np.clip(s, 0.0, 1.0)


def passage_cdf(s, x, theta, omega):
    """CDF of the first-passage time to omega for a fresh unit at stress x, evaluated at
    *virtual ages* s > 0 (for a unit switched at level b, pass w(b) + u). Vectorized in s."""
    a0, a1, lam, q = unpack(theta)
    s = np.asarray(s, dtype=float)
    mu = np.exp(a0 + a1 * x)
    a = mu * s**q
    b = lam * s ** (2.0 * q)
    return ig_survival(omega, a, b)  # P(Y_x(s) >= omega)


def passage_pdf(s, x, theta, omega):
    """First-passage density f_x(s) by central differences of passage_cdf (vectorized)."""
    s = np.asarray(s, dtype=float)
    h = np.maximum(1e-6, s * 1e-5)
    return (passage_cdf(s + h, x, theta, omega) - passage_cdf(np.maximum(s - h, 1e-12), x, theta, omega)) / (
        s + h - np.maximum(s - h, 1e-12)
    )


def virtual_age(b, x, theta):
    """CE virtual age w(b) at stress x for a unit switched with accumulated level b."""
    a0, a1, lam, q = unpack(theta)
    mu = np.exp(a0 + a1 * x)
    return (np.asarray(b, dtype=float) / mu) ** (1.0 / q)


def passage_quantile(p, x, theta, omega, t_hi=1e9):
    """Exact first-passage quantile at stress x by bisection on passage_cdf. Vectorized in p."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    t_lo = np.full_like(p, 1e-9)
    t_hi = np.full_like(p, float(t_hi))
    # shrink t_hi until CDF(t_hi) >= p (only where needed), then bisect
    for _ in range(80):
        s = passage_cdf(t_hi, x, theta, omega)
        need = s < p
        if not np.any(need):
            break
        t_hi = np.where(need, t_hi * 2.0, t_hi)
        t_hi = np.minimum(t_hi, 1e15)
    for _ in range(200):
        t_mid = 0.5 * (t_lo + t_hi)
        s = passage_cdf(t_mid, x, theta, omega)
        go_up = s < p
        t_lo = np.where(go_up, t_mid, t_lo)
        t_hi = np.where(go_up, t_hi, t_mid)
    return 0.5 * (t_lo + t_hi)


def xi_p_approx(p, x, theta, omega):
    """Closed-form p-quantile under the normal approximation of the first passage
    (Ye et al. 2014 / Ma et al. 2021 Eq. (5))."""
    from scipy.stats import norm

    a0, a1, lam, q = unpack(theta)
    mu = np.exp(a0 + a1 * x)
    z = norm.ppf(p)
    inner = z + np.sqrt(z * z + 4.0 * lam * omega / mu**2)
    return (mu / (4.0 * lam) * inner**2) ** (1.0 / q)


# ------------------------------------------------------------------ sampling

def sample_passage_time(rng, x, theta, omega, t_max=None):
    """Exact first-passage time at stress x by inversion; None if beyond t_max (censored)."""
    u = rng.random()
    if t_max is not None:
        f_max = float(passage_cdf(t_max, x, theta, omega))
        if u > f_max:
            return None
    t = float(passage_quantile(u, x, theta, omega)[0])
    return t


def sample_truncated_level(rng, x, theta, omega, tau):
    """Level B = Y_x(tau) of a survivor, by rejection from IG(a, b) truncated at omega."""
    a0, a1, lam, q = unpack(theta)
    mu = np.exp(a0 + a1 * x)
    a = mu * tau**q
    b = lam * tau ** (2 * q)
    out = omega + 1.0
    while out >= omega:
        out = rng.wald(a, b)
    return out


def sample_increment(rng, mean, shape):
    """One draw from IG(mean, shape)."""
    return rng.wald(mean, shape)
