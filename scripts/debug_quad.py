"""Debug the scheme-(b) quadrature: compare plain GL vs composite GL vs MC per u."""
import numpy as np

import src.ig_process as ig
import src.likelihood as lik
from src.likelihood import continuation_cdf, _continuation_pdf

theta = (-1.88, 1.73, 0.653, 0.449)
x0, x1, omega, tau1, tmax = 0.46, 0.78, 10.0, 2000.0, 5000.0
umax = tmax - tau1
a_base = np.exp(theta[0] + theta[1] * x0) * tau1 ** theta[3]
b_base = theta[2] * tau1 ** (2 * theta[3])
print(f"a_base={a_base:.3f} b_base={b_base:.1f} omega={omega}")

rng = np.random.default_rng(1)
m = 4_000_000
bs_all = rng.wald(a_base, b_base, size=m)
bs = bs_all[bs_all < omega]
print(f"MC truncation kept {len(bs)}/{m} = {len(bs)/m:.3f} (analytic "
      f"{1 - float(ig.ig_survival(omega, a_base, b_base)):.3f})")


def f_level(b):
    return np.exp(ig.log_ig_pdf(b, a_base, b_base))


def gl(n, lo=0.0, hi=None):
    hi = omega if hi is None else hi
    x, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * (x + 1) / 2, (hi - lo) * w / 2


def m_f_gl(u, n=160):
    b, w = gl(n)
    return float(np.sum(f_level(b) * _continuation_pdf(u, b, x1, theta, omega) * w))


def m_f_composite(u, panels=8, n=80):
    edges = np.linspace(0, omega, panels + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        b, w = gl(n, lo, hi)
        tot += float(np.sum(f_level(b) * _continuation_pdf(u, b, x1, theta, omega) * w))
    return tot


for u in [50.0, 500.0, 2000.0]:
    mc = float(np.mean(_continuation_pdf(u, bs, x1, theta, omega)))
    se = float(np.std(_continuation_pdf(u, bs, x1, theta, omega)) / np.sqrt(len(bs)))
    print(f"u={u:6.0f}: GL160={m_f_gl(u):.6f}  comp8x80={m_f_composite(u):.6f}  "
          f"comp32x60={m_f_composite(u, 32, 60):.6f}  MC={mc:.6f} (se={se:.2g})")

# integrand shape at the worst u
u = 50.0
b = np.linspace(0.05, omega - 0.05, 40)
vals = f_level(b) * _continuation_pdf(u, b, x1, theta, omega)
print("\nintegrand u=50 over b:")
for bb, vv in zip(b, vals):
    print(f"  b={bb:6.2f}  f_lev={f_level(bb):8.4f}  g={float(_continuation_pdf(u, bb, x1, theta, omega)):10.4f}  prod={vv:10.6f}")

mc_s = float(np.mean(1.0 - continuation_cdf(umax, bs, x1, theta, omega)))
print(f"\nm_s: GL160={float(np.sum(f_level(gl(160)[0]) * (1 - continuation_cdf(umax, gl(160)[0], x1, theta, omega)) * gl(160)[1])):.6f}  MC={mc_s:.6f}")
