"""Localize the design.py discrepancy: validate each information block separately."""
import numpy as np

import src.ig_process as ig
from src.design import _gl, vec_neg_hessian

THETA = np.array([-1.8944, 1.7377, 0.6285, 0.4491])
h = np.array([1e-4, 1e-4, 1e-6, 1e-6])
x0, x1, omega, tau1 = 0.46, 0.78, 10.0, 2000.0


def fd_neg_hess_scalar(f, th, hh):
    n = len(th)
    f0 = f(th)
    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n); ei[i] = hh[i]
        H[i, i] = -(f(th + ei) - 2 * f0 + f(th - ei)) / hh[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = hh[i]
            ej = np.zeros(n); ej[j] = hh[j]
            H[i, j] = H[j, i] = -(f(th + ei + ej) - f(th + ei - ej) - f(th - ei + ej) + f(th - ei - ej)) / (4 * hh[i] * hh[j])
    return H


print("=== block 1: baseline observation B ~ IG(a0(tau1), b0(tau1)) ===")
a_base = np.exp(THETA[0] + THETA[1] * x0) * tau1 ** THETA[3]
b_base = THETA[2] * tau1 ** (2 * THETA[3])
rng = np.random.default_rng(1)
m = 400_000
bs = rng.wald(a_base, b_base, size=m)
bs = bs[bs < omega]
emp = np.zeros((4, 4))
for b in bs[:4000]:
    emp += fd_neg_hess_scalar(lambda th, b=b: float(ig.log_ig_pdf(b, np.exp(th[0] + th[1] * x0) * tau1 ** th[3], th[2] * tau1 ** (2 * th[3]))), THETA, h)
emp = emp.mean(axis=None) / 1  # placeholder
emp = np.zeros((4, 4))
cnt = 0
for b in bs[:4000]:
    emp += fd_neg_hess_scalar(lambda th, b=b: float(ig.log_ig_pdf(b, np.exp(th[0] + th[1] * x0) * tau1 ** th[3], th[2] * tau1 ** (2 * th[3]))), THETA, h)
    cnt += 1
emp /= cnt

nodes, wt = _gl(96, 0.0, omega)
fB = np.exp(ig.log_ig_pdf(nodes, a_base, b_base))
ana = np.einsum("ijn,jn->ij", vec_neg_hessian(
    lambda z, bb: ig.log_ig_pdf(bb, np.exp(z[0] + z[1] * x0) * tau1 ** z[3], z[2] * tau1 ** (2 * z[3])),
    THETA, h, nodes), fB * wt[None, :])
print("analytic:\n", np.round(ana, 4))
print("empirical (truncated MC):\n", np.round(emp, 4))

print("\n=== block 2: continuation increment | b ===")
b_val = 8.0
u1, u2 = 0.0, 300.0
w = (b_val / np.exp(THETA[0] + THETA[1] * x1)) ** (1 / THETA[3])
dlam = (w + u2) ** THETA[3] - (w + u1) ** THETA[3]
a_int = np.exp(THETA[0] + THETA[1] * x1) * dlam
s_int = THETA[2] * dlam**2
draws = rng.wald(a_int, s_int, size=4000)


def log_inc(th):
    ww = (b_val / np.exp(th[0] + th[1] * x1)) ** (1 / th[3])
    dl = (ww + u2) ** th[3] - (ww + u1) ** th[3]
    return float(ig.log_ig_pdf(draws[len(draws) // 2], np.exp(th[0] + th[1] * x1) * dl, th[2] * dl**2))


emp2 = np.zeros((4, 4))
for d in draws:
    def f_one(th, d=d):
        ww = (b_val / np.exp(th[0] + th[1] * x1)) ** (1 / th[3])
        dl = (ww + u2) ** th[3] - (ww + u1) ** th[3]
        return float(ig.log_ig_pdf(d, np.exp(th[0] + th[1] * x1) * dl, th[2] * dl**2))
    emp2 += fd_neg_hess_scalar(f_one, THETA, h)
emp2 /= len(draws)
print("empirical mean -Hessian of log f_inc:\n", np.round(emp2, 4))

# analytic via J' W J
def inc_params(z):
    ww = (b_val / np.exp(z[0] + z[1] * x1)) ** (1 / z[3])
    dl = (ww + u2) ** z[3] - (ww + u1) ** z[3]
    return np.exp(z[0] + z[1] * x1) * dl, z[2] * dl**2


J = np.zeros((4, 2))
for k in range(4):
    tp = THETA.copy(); tm = THETA.copy()
    tp[k] += h[k]; tm[k] -= h[k]
    ap, sp = inc_params(tp)
    am, sm = inc_params(tm)
    J[k, 0] = (ap - am) / (2 * h[k])
    J[k, 1] = (sp - sm) / (2 * h[k])
W = np.diag([s_int / a_int**3, 1 / (2 * s_int**2)])
ana2 = J @ W @ J.T  # J[k, m] = d param_m / d theta_k  =>  I_theta = J W J'
print("analytic J W J':\n", np.round(ana2, 4))
print("ratio emp/ana:\n", np.round(emp2 / ana2, 3))
