"""Fit the constant-stress IG ADT model to the electrical-connector stress-relaxation data
(Meeker & Escobar / Yang 2007 Example 8.7; as tabulated in Wang et al. 2016 and Ma et al. 2021)
and compare with the published MLEs of Ma et al. (2021):
    (alpha0, alpha1, lam, q) = (-1.88, 1.73, 0.653, 0.449),  Lambda(t) = t^q,  omega = 30.

Standardization (Arrhenius): x = (1/s0 - 1/s) / (1/s0 - 1/sH), s in Kelvin,
s0 = 313.15 K (40 C), sH = 373.15 K (100 C).
"""
import numpy as np
from scipy.optimize import minimize

import src.ig_process as ig

T65 = [108, 241, 534, 839, 1074, 1350, 1637, 1890, 2178, 2513, 2810]
T85 = [46, 108, 212, 408, 632, 764, 1011, 1333, 1517, 2586]
T100 = [46, 108, 212, 344, 446, 626, 729, 972, 1005, 1218]

# Wang et al. (2016) Table 1 (unit 2 at 65C: 7th value replaced by mean(6.85, 7.4) = 7.125)
DATA = {
    65.0: [
        [2.12, 2.7, 3.52, 4.25, 5.55, 6.12, 6.75, 7.22, 7.68, 8.46, 9.46],
        [2.29, 3.24, 4.16, 4.86, 5.74, 6.85, 7.125, 7.4, 8.14, 9.25, 10.55],
        [2.4, 3.61, 4.35, 5.09, 5.5, 7.03, 8.24, 8.81, 9.629, 10.27, 11.11],
        [2.31, 3.48, 5.51, 6.2, 7.31, 7.96, 8.57, 9.07, 10.46, 11.48, 12.31],
        [3.14, 4.33, 5.92, 7.22, 8.14, 9.07, 9.44, 10.09, 11.2, 12.77, 13.51],
        [3.59, 5.55, 5.92, 7.68, 8.61, 10.37, 11.11, 12.22, 13.51, 14.16, 15.0],
    ],
    85.0: [
        [2.77, 4.62, 5.83, 6.66, 8.05, 10.61, 11.2, 11.98, 13.33, 15.64],
        [3.88, 4.37, 6.29, 7.77, 9.16, 9.9, 10.37, 12.77, 14.72, 16.8],
        [3.18, 4.53, 6.94, 8.14, 8.79, 10.09, 11.11, 14.72, 16.47, 18.66],
        [3.61, 4.37, 6.29, 7.87, 9.35, 11.48, 12.4, 13.7, 15.37, 18.51],
        [3.42, 4.25, 7.31, 8.61, 10.18, 12.03, 13.7, 15.27, 17.22, 19.25],
        [5.27, 5.92, 8.05, 9.81, 12.4, 13.24, 15.83, 17.59, 20.09, 23.51],
    ],
    100.0: [
        [4.25, 5.18, 8.33, 9.53, 11.48, 13.14, 15.55, 16.94, 18.05, 19.44],
        [4.81, 6.16, 7.68, 9.25, 10.37, 12.4, 15.0, 16.2, 18.24, 20.09],
        [5.09, 7.03, 8.33, 10.37, 12.22, 14.35, 16.11, 18.7, 19.72, 21.66],
        [4.81, 7.5, 9.16, 10.55, 13.51, 15.55, 16.57, 19.07, 20.27, 22.4],
        [5.64, 6.57, 8.61, 12.5, 14.44, 16.57, 18.7, 21.2, 22.59, 24.07],
        [4.72, 8.14, 10.18, 12.4, 15.09, 17.22, 19.16, 21.57, 24.35, 26.2],
    ],
}
TIMES = {65.0: T65, 85.0: T85, 100.0: T100}


def x_of_temp(temp_c):
    s0, sh = 313.15, 373.15
    s = temp_c + 273.15
    return (1.0 / s0 - 1.0 / s) / (1.0 / s0 - 1.0 / sh)


def nll(z):
    a0, a1, lam, q = z[0], z[1], np.exp(z[2]), np.exp(z[3])
    total = 0.0
    for temp, units in DATA.items():
        x = x_of_temp(temp)
        mu = np.exp(a0 + a1 * x)
        ts = np.asarray(TIMES[temp], float)
        dlam = np.diff(np.concatenate([[0.0], ts]) ** q)  # t_j^q - t_{j-1}^q
        for lev in units:
            lev = np.asarray(lev, float)
            inc = np.diff(np.concatenate([[0.0], lev]))
            total -= np.sum(ig.log_ig_pdf(inc, mu * dlam, lam * dlam**2))
    if not np.isfinite(total):
        return 1e12
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    best = None
    for k in range(12):
        z0 = np.array([-1.88, 1.73, np.log(0.653), np.log(0.449)]) if k == 0 else np.array(
            [rng.uniform(-4, 1), rng.uniform(0, 4), rng.uniform(-1, 1), rng.uniform(-1.5, 0)]
        )
        res = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        res = minimize(nll, res.x, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-11})
        if best is None or res.fun < best.fun:
            best = res
    th = np.array([best.x[0], best.x[1], np.exp(best.x[2]), np.exp(best.x[3])])
    print("our fit      (a0, a1, lam, q) =", np.round(th, 4), " nll:", round(float(best.fun), 3))
    print("Ma et al.    (a0, a1, lam, q) = (-1.88, 1.73, 0.653, 0.449)")
    for t in [65.0, 85.0, 100.0]:
        print(f"x({t} C) = {x_of_temp(t):.4f}")
    print("xi_0.1 at 40C (ours):", round(float(ig.xi_p_approx(0.1, 0.0, th, 30.0)), 1))
    print("xi_0.1 at 40C (Ma):  ~ via their params:", round(float(ig.xi_p_approx(0.1, 0.0, (-1.88, 1.73, 0.653, 0.449), 30.0)), 1))
