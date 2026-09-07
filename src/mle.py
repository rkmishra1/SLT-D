"""MLE for the SLT-D model: multi-start optimization over (alpha0, alpha1, log lam, log q),
observed-information covariance, and delta-method inference for log xi_p at use condition.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import src.ig_process as ig
import src.likelihood as lik

# unconstrained parameterization z = (alpha0, alpha1, log lam, log q)
Z_SCALE = np.array([1.0, 1.0, 0.25, 0.25])
# soft box to keep multi-start away from degenerate flat-likelihood corners
Z_LO = np.array([-12.0, -2.0, -9.0, np.log(0.05)])
Z_HI = np.array([4.0, 12.0, 9.0, np.log(3.0)])


def _to_theta(z):
    return np.array([z[0], z[1], np.exp(z[2]), np.exp(z[3])])


def _to_z(theta):
    return np.array([theta[0], theta[1], np.log(theta[2]), np.log(theta[3])])


def _in_box(z):
    return np.all(z >= Z_LO) and np.all(z <= Z_HI)


def fit_mle(world, scheme, z_starts, maxiter=400):
    """Multi-start Nelder-Mead on the negative log-likelihood. Returns dict or None on failure."""

    def nll(z):
        if not _in_box(z):
            return 1e12
        try:
            v = -lik.loglik(scheme, _to_theta(z), world)
        except (FloatingPointError, ValueError):
            return 1e12
        if not np.isfinite(v):
            return 1e12
        return v

    best = None
    for z0 in z_starts:
        try:
            res = minimize(nll, z0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8})
            res = minimize(nll, res.x, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-9})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        return None
    theta_hat = _to_theta(best.x)
    return {"z": best.x, "theta": theta_hat, "nll": float(best.fun), "ok": True}


def observed_info_cov(z_hat, scheme, world, h=None):
    """Covariance of z-hat from the numerical Hessian of the negative log-likelihood.

    FD steps must stay well above the nested-differentiation noise floor: the log-likelihood
    contains numerical t-derivatives (passage densities), so h^2 >> 1e-12 is required.
    """
    if h is None:
        h = np.array([1e-3, 1e-3, 1e-3, 1e-4])
    n = len(z_hat)
    H = np.zeros((n, n))
    f = lambda z: -lik.loglik(scheme, _to_theta(z), world)

    def safe(z):
        v = f(z)
        return v if np.isfinite(v) else 1e12

    f0 = safe(z_hat)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (safe(z_hat + ei) - 2 * f0 + safe(z_hat - ei)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (safe(z_hat + ei + ej) - safe(z_hat + ei - ej)
                                 - safe(z_hat - ei + ej) + safe(z_hat - ei - ej)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
            return None
        return 0.5 * (cov + cov.T)
    except np.linalg.LinAlgError:
        return None


def cov_theta_from_z(cov_z, theta):
    """Transform the z-space covariance (alpha0, alpha1, log lam, log q) to theta space:
    dtheta/dz = diag(1, 1, lam, q)."""
    lam, q = theta[2], theta[3]
    D = np.diag([1.0, 1.0, lam, q])
    return D @ cov_z @ D


def grad_log_xi_p(p, x, theta, omega, h=None):
    """Gradient of log xi_p(x) (closed-form quantile) wrt theta, by central differences."""
    theta = np.asarray(theta, float)
    if h is None:
        h = np.array([1e-5, 1e-5, 1e-6, 1e-6])
    g = np.zeros(4)
    for i in range(4):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += h[i]
        tm[i] -= h[i]
        g[i] = (np.log(ig.xi_p_approx(p, x, tp, omega)) - np.log(ig.xi_p_approx(p, x, tm, omega))) / (2 * h[i])
    return g


def xi_inference(fit, scheme, world, p=0.1, x_use=0.0, omega=30.0):
    """Point estimate, SE and 95% CI for log xi_p at use condition.

    The Hessian is taken in z-space (numerically stable); the covariance is transformed to
    theta space before the delta method so that the gradient g = d log xi / d theta and the
    covariance live in the same coordinates.
    """
    theta_hat = fit["theta"]
    log_xi = float(np.log(ig.xi_p_approx(p, x_use, theta_hat, omega)))
    cov_z = observed_info_cov(fit["z"], scheme, world)
    if cov_z is None:
        return {"log_xi": log_xi, "se": None, "lo": None, "hi": None}
    cov_t = cov_theta_from_z(cov_z, theta_hat)
    g = grad_log_xi_p(p, x_use, theta_hat, omega)
    var = float(g @ cov_t @ g)
    if var <= 0 or not np.isfinite(var):
        return {"log_xi": log_xi, "se": None, "lo": None, "hi": None}
    se = np.sqrt(var)
    from scipy.stats import norm

    cr = norm.ppf(0.975)
    return {"log_xi": log_xi, "se": se, "lo": log_xi - cr * se, "hi": log_xi + cr * se}


def make_starts(theta0, n_starts=6, seed=0, scale=0.12):
    """Jittered starts around planning values + one generic start."""
    rng = np.random.default_rng(seed)
    z0 = _to_z(theta0)
    starts = [z0 + rng.normal(0, scale, 4) * Z_SCALE for _ in range(n_starts)]
    starts.append(np.array([-1.0, 1.0, np.log(0.5), np.log(0.5)]))
    return starts
