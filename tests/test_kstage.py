import unittest

import numpy as np

import src.ig_process as ig
import src.likelihood as lik
import src.kstage as ks
import src.mle as mle
from src.simulate import simulate_world

THETA = (-1.8944, 1.7377, 0.6285, 0.4491)


def design_k(stages, read_step=50.0, n=12, x0=0.46, omega=10.0, tmax=5000.0):
    tmax_last = stages[-1][0]
    return {
        "x0": x0, "omega": omega, "n": n, "tmax": tmax, "stages": stages,
        "read_offsets": np.arange(read_step, tmax - tmax_last + 1.0, read_step),
    }


class TestKStage(unittest.TestCase):
    def test_structure(self):
        rng = np.random.default_rng(3)
        d = design_k([(1500.0, 0.3, 0.78), (3000.0, 0.4, 1.0)])
        w = ks.simulate_world_kstage(rng, np.array(THETA), d)
        n_withdrawn = sum(len(c["units"]) for c in w["cohorts"])
        self.assertEqual(len(w["failures"]) + n_withdrawn + w["n_cens"], d["n"])
        for c in w["cohorts"]:
            for u in c["units"]:
                self.assertLess(u["B"], d["omega"])
                self.assertTrue(np.all(np.diff(u["read_levels"]) > 0) or len(u["read_levels"]) == 1)
        # taus ascending, withdrawals only from survivors
        taus = [c["tau"] for c in w["cohorts"]]
        self.assertEqual(taus, sorted(taus))

    def test_k1_equivalence_with_two_stage(self):
        """K=1 k-stage likelihood must equal the 2-stage scheme-(c) likelihood on the same data."""
        rng = np.random.default_rng(17)
        d2 = {"x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 12, "tau1": 2000.0, "tmax": 5000.0,
              "pi1": 0.5, "read_offsets": np.arange(50.0, 3001.0, 50.0)}
        w2 = simulate_world(rng, np.array(THETA), d2)
        # convert to k-stage world
        failures = list(w2["fail_pre"]) + [t for t in w2["stayers"] if t is not None]
        n_cens = sum(1 for t in w2["stayers"] if t is None)
        coh = [{"tau": d2["tau1"], "x": d2["x1"], "units": w2["withdrawn"]}]
        wk = {"design": design_k([(2000.0, 0.5, 0.78)]), "failures": np.sort(np.array(failures)),
              "n_cens": n_cens, "cohorts": coh}
        for th in [THETA, (-1.5, 1.9, 0.8, 0.5)]:
            v2 = lik.loglik("c", np.array(th), w2)
            vk = ks.loglik_c_kstage(np.array(th), wk)
            self.assertAlmostEqual(v2, vk, places=8, msg=f"theta={th}: {v2} vs {vk}")

    def test_score_identity(self):
        rng = np.random.default_rng(21)
        d = design_k([(1500.0, 0.3, 0.78), (3000.0, 0.4, 1.0)])
        reps, h = 300, np.array([2e-4, 2e-4, 2e-6, 2e-6]) * np.array([1.0, 1.0, 0.01, 0.01])
        S = []
        for _ in range(reps):
            w = ks.simulate_world_kstage(rng, np.array(THETA), d)
            s = np.zeros(4)
            for i in range(4):
                tp = np.array(THETA, float)
                tm = np.array(THETA, float)
                tp[i] += h[i]
                tm[i] -= h[i]
                s[i] = (ks.loglik_c_kstage(tp, w) - ks.loglik_c_kstage(tm, w)) / (2 * h[i])
            S.append(s)
        S = np.array(S)
        mean, se = S.mean(axis=0), S.std(axis=0) / np.sqrt(reps)
        for i in range(4):
            self.assertLess(abs(mean[i]), 4.0 * se[i] + 0.02, msg=f"score[{i}]={mean[i]:.3f}, se={se[i]:.3f}")

    def test_mle_recovery_k2(self):
        rng = np.random.default_rng(55)
        d = design_k([(1500.0, 0.3, 0.78), (3000.0, 0.4, 1.0)], n=30)
        errs = []
        for rep in range(6):
            w = ks.simulate_world_kstage(rng, np.array(THETA), d)
            starts = mle.make_starts(THETA, n_starts=4, seed=rep)

            def nll(z):
                th = np.array([z[0], z[1], np.exp(z[2]), np.exp(z[3])])
                v = -ks.loglik_c_kstage(th, w)
                return v if np.isfinite(v) else 1e12

            from scipy.optimize import minimize

            best = None
            for z0 in starts:
                r = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 300})
                if best is None or r.fun < best.fun:
                    best = r
            th = np.array([best.x[0], best.x[1], np.exp(best.x[2]), np.exp(best.x[3])])
            errs.append(np.log(ig.xi_p_approx(0.1, 0.0, th, 10.0)))
        log_xi_true = np.log(ig.xi_p_approx(0.1, 0.0, np.array(THETA), 10.0))
        err = np.abs(np.array(errs) - log_xi_true)
        self.assertLess(np.median(err), 0.8, msg=f"errors={np.round(err, 3)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
