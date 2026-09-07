"""Per-rep inspection of scheme-(c) fits: convergence quality and lambda behavior."""
import numpy as np

import src.ig_process as ig
import src.mle as mle
import src.likelihood as lik
from src.simulate import simulate_world

theta = (-1.88, 1.73, 0.653, 0.449)
design = {
    "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 20, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
    "read_offsets": np.arange(50.0, 3001.0, 50.0),
}
log_xi_true = float(np.log(ig.xi_p_approx(0.1, 0.0, theta, 10.0)))
print("true:", theta, " log xi0.1:", round(log_xi_true, 3))

rng = np.random.default_rng(4242)
rows = []
for rep in range(15):
    w = simulate_world(rng, theta, design)
    starts = mle.make_starts(theta, n_starts=5, seed=rep)
    f = mle.fit_mle(w, "c", starts)
    # loglik at truth for comparison
    ll_true = lik.loglik("c", np.array(theta), w)
    inf = mle.xi_inference(f, "c", w, 0.1, 0.0, 10.0)
    rows.append((f["theta"], f["nll"], -ll_true, inf))
    print(f"rep{rep:02d} d1={w['d1']:2d} r1={w['r1']:2d}  theta=({f['theta'][0]:7.3f},{f['theta'][1]:6.3f},"
          f"{f['theta'][2]:7.4f},{f['theta'][3]:6.3f})  nll={f['nll']:9.3f} vs truth {-ll_true:9.3f}"
          f"  log_xi={inf['log_xi']:7.3f} se={inf['se'] if inf['se'] is None else round(inf['se'],3)}")

th = np.array([r[0] for r in rows])
print("\nmedian theta:", np.round(np.median(th, axis=0), 4))
print("mean   theta:", np.round(th.mean(axis=0), 4))
lxi = np.array([r[3]["log_xi"] for r in rows])
print("median log_xi:", round(float(np.median(lxi)), 3), " RMedSE-ish |err|:", round(float(np.median(np.abs(lxi - log_xi_true))), 3))
print("RMSE log_xi:", round(float(np.sqrt(np.mean((lxi - log_xi_true) ** 2))), 3))
