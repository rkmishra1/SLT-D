import unittest

import numpy as np

import src.ig_process as ig
import src.mle as mle
import src.likelihood as lik
from src.simulate import simulate_world

THETA = (-1.8944, 1.7377, 0.6285, 0.4491)


def make_design():
    return {
        "x0": 0.46, "x1": 0.78, "omega": 10.0, "n": 12, "tau1": 2000.0, "tmax": 5000.0, "pi1": 0.5,
        "read_offsets": np.arange(50.0, 3001.0, 50.0),
    }


class TestVariantB(unittest.TestCase):
    def test_variant_b_below_variant_a_in_expectation(self):
        """Marginalizing over b cannot add information: E[ell_B] <= E[ell_A] is not required
        pointwise, but ell_B(theta) = log Int f(b, data)db relates to ell_A by Jensen:
        ell_A uses the true b, so per unit ell_A - ell_B should average >= 0."""
        rng = np.random.default_rng(8)
        design = make_design()
        diffs = []
        for _ in range(80):
            w = simulate_world(rng, np.array(THETA), design)
            la = lik.loglik("c", np.array(THETA), w)
            lb = lik.loglik("c_nb", np.array(THETA), w)
            diffs.append(la - lb)
        self.assertGreater(np.mean(diffs), 0.0)

    def test_variant_b_score_identity(self):
        rng = np.random.default_rng(21)
        design = make_design()
        reps = 300
        h = np.array([2e-4, 2e-4, 2e-6, 2e-6]) * np.array([1.0, 1.0, 0.01, 0.01])
        S = []
        for _ in range(reps):
            w = simulate_world(rng, np.array(THETA), design)
            s = np.zeros(4)
            for i in range(4):
                tp = np.array(THETA, float)
                tm = np.array(THETA, float)
                tp[i] += h[i]
                tm[i] -= h[i]
                s[i] = (lik.loglik("c_nb", tp, w) - lik.loglik("c_nb", tm, w)) / (2 * h[i])
            S.append(s)
        S = np.array(S)
        mean, se = S.mean(axis=0), S.std(axis=0) / np.sqrt(reps)
        for i in range(4):
            self.assertLess(abs(mean[i]), 4.0 * se[i] + 0.02, msg=f"score[{i}]={mean[i]:.3f}, se={se[i]:.3f}")

    def test_variant_b_mle_close_to_variant_a(self):
        """Losing the baseline reading should cost precision but not consistency."""
        rng = np.random.default_rng(55)
        design = {**make_design(), "n": 30}
        est_a, est_b = [], []
        for rep in range(6):
            w = simulate_world(rng, np.array(THETA), design)
            starts = mle.make_starts(THETA, n_starts=4, seed=rep)
            fa = mle.fit_mle(w, "c", starts)
            fb = mle.fit_mle(w, "c_nb", starts)
            est_a.append(np.log(ig.xi_p_approx(0.1, 0.0, fa["theta"], 10.0)))
            est_b.append(np.log(ig.xi_p_approx(0.1, 0.0, fb["theta"], 10.0)))
        log_xi_true = np.log(ig.xi_p_approx(0.1, 0.0, np.array(THETA), 10.0))
        err_a = np.abs(np.array(est_a) - log_xi_true)
        err_b = np.abs(np.array(est_b) - log_xi_true)
        # variant B should stay within a factor ~3 of variant A's error on average
        self.assertLess(np.mean(err_b), 3.0 * max(np.mean(err_a), 0.05) + 0.5,
                        msg=f"errA={err_a}, errB={err_b}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
