"""Block 3 diagnostic: stage-0 failure/censoring information, analytic quadrature vs MC."""
import numpy as np

import src.ig_process as ig
from src.design import _gl, vec_neg_hessian, hessian_fd

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
h = np.array([1e-4, 1e-4, 1e-6, 1e-6])
x0, omega, tau1, tmax, pi1 = 0.46, 10.0, 2000.0, 5000.0, 0.5


def logf0(th, t):
    return np.log(np.maximum(ig.passage_pdf(t, x0, th, omega), 1e-300))


# --- analytic (as in design.py)
I_ana = np.zeros((4, 4))
for lo, hi in [(1e-6, tau1), (tau1, tmax)]:
    t, wt = _gl(48, lo, hi)
    Hn = vec_neg_hessian(lambda z, tt: logf0(z, tt), THETA, h, t)
    dens = ig.passage_pdf(t, x0, THETA, omega)
    I_ana += np.einsum("ijn,jn->ij", Hn, dens * wt[None, :]) * (1.0 if lo == 1e-6 else (1.0 - pi1))

HS = hessian_fd(lambda z: np.log(max(1.0 - float(ig.passage_cdf(tmax, x0, z, omega)), 1e-300)), THETA, h)
I_ana += (1.0 - pi1) * float(1.0 - ig.passage_cdf(tmax, x0, THETA, omega)) * HS


# --- MC: sample unit fates exactly
def mc_fail_info(m=40_000, seed=7):
    rng = np.random.default_rng(seed)
    acc = np.zeros((4, 4))

    def H_at(th, t):
        return vec_neg_hessian(lambda z, tt: logf0(z, tt), th, h, np.atleast_1d(t))[:, :, 0]

    n_fail_pre = 0
    n_fail_post = 0
    n_cens = 0
    for _ in range(m):
        u = rng.random()
        # sample T by inversion on a coarse then fine bisection
        lo, hi = 1e-9, 1e9
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(ig.passage_cdf(mid, x0, THETA, omega)) < u:
                lo = mid
            else:
                hi = mid
        T = 0.5 * (lo + hi)
        if T <= tau1:
            acc += H_at(THETA, T)
            n_fail_pre += 1
        else:
            if rng.random() < pi1:
                continue  # withdrawn: not part of the fail block
            if T <= tmax:
                acc += H_at(THETA, T)
                n_fail_post += 1
            else:
                acc += HS
                n_cens += 1
    return acc / m, n_fail_pre / m, n_fail_post / m, n_cens / m


I_mc, f_pre, f_post, f_cens = mc_fail_info(m=30_000)
F1 = float(ig.passage_cdf(tau1, x0, THETA, omega))
F2 = float(ig.passage_cdf(tmax, x0, THETA, omega))
print(f"MC fate probs: pre={f_pre:.4f} (F(tau1)={F1:.4f})  post={f_post:.4f} "
      f"(expect {(F2 - F1) * (1 - pi1):.4f})  cens={f_cens:.4f} (expect {(1 - F2) * (1 - pi1):.4f})")
print("analytic:\n", np.round(I_ana, 4))
print("MC:\n", np.round(I_mc, 4))
print("ratio MC/ana:\n", np.round(I_mc / I_ana, 3))
