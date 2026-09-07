"""Smoke test: MLE recovery per scheme + timing."""
import time

import numpy as np

import src.ig_process as ig
import src.mle as mle
from src.simulate import simulate_world

theta = (-1.88, 1.73, 0.653, 0.449)
design = {
    "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 20, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
    "read_offsets": np.arange(50.0, 3001.0, 50.0),
}
print("true theta:", theta)
print("true log xi_0.1(x=0):", np.log(ig.xi_p_approx(0.1, 0.0, theta, 10.0)))

rng = np.random.default_rng(42)
for scheme in ["a", "b", "c"]:
    t0 = time.time()
    fits = []
    for rep in range(8):
        w = simulate_world(rng, theta, design)
        fit = mle.fit_mle(w, scheme, mle.make_starts(theta, n_starts=4, seed=rep))
        fits.append(fit)
    dt = (time.time() - t0) / 8
    th = np.array([f["theta"] for f in fits])
    print(f"\nscheme {scheme}: {dt:.2f} s/fit")
    print("  mean theta_hat:", np.round(th.mean(axis=0), 4))
    print("  sd   theta_hat:", np.round(th.std(axis=0), 4))
    lx = [mle.xi_inference(f, scheme, w, 0.1, 0.0, 10.0) for f, w in
          zip(fits, [None] * len(fits))] if False else None
    # coverage-ish: just show one CI
    w = simulate_world(np.random.default_rng(999), theta, design)
    f = mle.fit_mle(w, scheme, mle.make_starts(theta, n_starts=4, seed=1))
    inf = mle.xi_inference(f, scheme, w, 0.1, 0.0, 10.0)
    print("  one fit log_xi:", round(inf["log_xi"], 3), " se:", None if inf["se"] is None else round(inf["se"], 3),
          " CI:", None if inf["lo"] is None else (round(inf["lo"], 3), round(inf["hi"], 3)))
