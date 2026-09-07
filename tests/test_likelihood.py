import unittest

import numpy as np

import src.ig_process as ig
import src.likelihood as lik
from src.simulate import simulate_world


def make_design(x0=0.46, x1=0.78, omega=10.0, n=12, tau1=2000.0, tmax=5000.0, pi1=0.5, step=50.0):
    """Scenario with a substantial failure fraction pre-tau1 (exercises all blocks)."""
    return {
        "x0": x0,
        "x1": x1,
        "omega": omega,
        "n": n,
        "tau1": tau1,
        "tmax": tmax,
        "pi1": pi1,
        "read_offsets": np.arange(step, tmax - tau1 + 1.0, step),
    }


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.theta = (-1.88, 1.73, 0.653, 0.449)
        self.design = make_design(x0=0.78, x1=1.0, omega=30.0)

    def test_world_structure(self):
        rng = np.random.default_rng(3)
        w = simulate_world(rng, self.theta, self.design)
        self.assertEqual(w["d1"] + w["r1"] + len(w["stayers"]), self.design["n"])
        self.assertEqual(w["r1"], int(np.floor(self.design["pi1"] * (self.design["n"] - w["d1"]))))
        for t in w["fail_pre"]:
            self.assertLessEqual(t, self.design["tau1"])
        for t in w["stayers"]:
            if t is not None:
                self.assertGreater(t, self.design["tau1"])
        for k in w["withdrawn"]:
            self.assertLess(k["B"], self.design["omega"])

    def test_failure_rate_matches_cdf(self):
        # fraction of units failing before tau1 across many worlds ~ F_{x0}(tau1)
        rng = np.random.default_rng(5)
        n, reps = 60, 400
        fails = 0
        for _ in range(reps):
            w = simulate_world(rng, self.theta, {**self.design, "n": n, "omega": 10.0, "x0": 0.46})
            fails += w["d1"]
        emp = fails / (n * reps)
        an = float(ig.passage_cdf(self.design["tau1"], 0.46, self.theta, 10.0))
        self.assertAlmostEqual(emp, an, delta=4.0 * np.sqrt(an * (1 - an) / (n * reps)) + 2e-3)

    def test_continuation_fail_rate_matches_G(self):
        # MC check of the scheme-(b) simulator against continuation_cdf
        rng = np.random.default_rng(9)
        theta, d = self.theta, self.design
        x1, omega, tau1, tmax = d["x1"], d["omega"], d["tau1"], d["tmax"]
        umax = tmax - tau1
        m = 4000
        bs = np.array([ig.sample_truncated_level(rng, d["x0"], theta, omega, tau1) for _ in range(m)])
        crossed = 0
        from src.likelihood import continuation_cdf

        for b in bs:
            uu = rng.random()
            if uu <= float(continuation_cdf(umax, b, x1, theta, omega)):
                crossed += 1
        # analytic marginal crossing probability (integrate G over the latent level density)
        bn, bw = lik._gl_nodes(200)
        b_nodes = omega * bn
        f_b = np.exp(ig.log_ig_pdf(b_nodes, np.exp(theta[0] + theta[1] * d["x0"]) * tau1**theta[3],
                                   theta[2] * tau1 ** (2 * theta[3])))
        an = float(np.sum(f_b * continuation_cdf(umax, b_nodes, x1, theta, omega) * bw * omega))
        self.assertAlmostEqual(crossed / m, an, delta=4.0 * np.sqrt(an * (1 - an) / m) + 1e-3)

    def test_quadrature_mf_ms_against_mc(self):
        # scheme-(b) Gauss-Legendre integrals vs Monte Carlo integration.
        # MC draws are from the level density truncated at omega, so E_mc[g] = m_f / A
        # with A = P(B < omega): multiply the MC mean by A.
        rng = np.random.default_rng(31)
        theta = self.theta
        d = make_design(x0=0.46, x1=0.78, omega=10.0)
        x0, x1, omega, tau1, tmax = d["x0"], d["x1"], d["omega"], d["tau1"], d["tmax"]
        umax = tmax - tau1
        m = 2_000_000
        a_base = np.exp(theta[0] + theta[1] * x0) * tau1**theta[3]
        b_base = theta[2] * tau1 ** (2 * theta[3])
        bs = rng.wald(a_base, b_base, size=m)
        A_emp = float(np.mean(bs < omega))
        bs = bs[bs < omega]
        from src.likelihood import continuation_cdf, _continuation_pdf

        for u in [50.0, 500.0, 2000.0]:
            nodes, wts = lik._gl_nodes(160)
            f_b = np.exp(ig.log_ig_pdf(omega * nodes, a_base, b_base))
            gl = float(np.sum(f_b * _continuation_pdf(u, omega * nodes, x1, theta, omega) * (wts * omega)))
            mc = float(np.mean(_continuation_pdf(u, bs, x1, theta, omega))) * A_emp
            self.assertAlmostEqual(gl, mc, delta=0.05 * mc + 1e-6)
        gl_s = float(np.sum(f_b * (1.0 - continuation_cdf(umax, omega * nodes, x1, theta, omega)) * (wts * omega)))
        mc_s = float(np.mean(1.0 - continuation_cdf(umax, bs, x1, theta, omega))) * A_emp
        self.assertAlmostEqual(gl_s, mc_s, delta=0.05 * mc_s + 1e-6)


class TestScoreAndPerturbation(unittest.TestCase):
    """E_theta[grad ell] = 0 (within MC noise) and theta_true maximizes E[ell] locally."""

    def setUp(self):
        self.theta = (-1.88, 1.73, 0.653, 0.449)
        self.design = make_design()

    def _score_of(self, scheme, w, h):
        s = np.zeros(4)
        for i in range(4):
            tp = np.array(self.theta, float)
            tm = np.array(self.theta, float)
            tp[i] += h[i]
            tm[i] -= h[i]
            s[i] = (lik.loglik(scheme, tp, w) - lik.loglik(scheme, tm, w)) / (2 * h[i])
        return s

    def _check_scheme(self, scheme):
        rng = np.random.default_rng(21)
        reps = 400
        h = np.array([2e-4, 2e-4, 2e-6, 2e-6]) * np.array([1.0, 1.0, 0.01, 0.01])
        S = np.array([self._score_of(scheme, w, h) for w in
                      (simulate_world(rng, self.theta, self.design) for _ in range(reps))])
        mean, se = S.mean(axis=0), S.std(axis=0) / np.sqrt(reps)
        for i in range(4):
            self.assertLess(abs(mean[i]), 4.0 * se[i] + 0.02,
                            msg=f"{scheme}: score[{i}]={mean[i]:.3f}, se={se[i]:.3f}, full={mean}")

    def test_scheme_a(self):
        self._check_scheme("a")

    def test_scheme_b(self):
        self._check_scheme("b")

    def test_scheme_c(self):
        self._check_scheme("c")

    def test_paired_perturbation_all_schemes(self):
        # CRN-paired: mean[ell(theta) - ell(theta +/- h)] should be >= 0 (concavity at the max)
        rng = np.random.default_rng(77)
        reps = 300
        worlds = [simulate_world(rng, self.theta, self.design) for _ in range(reps)]
        hs = [0.2, 0.2, 0.1, 0.05]
        for scheme in ["a", "b", "c"]:
            for i, hh in enumerate(hs):
                tp = np.array(self.theta, float)
                tm = np.array(self.theta, float)
                tp[i] += hh
                tm[i] -= hh
                dp = np.mean([lik.loglik(scheme, self.theta, w) - lik.loglik(scheme, tp, w) for w in worlds])
                dm = np.mean([lik.loglik(scheme, self.theta, w) - lik.loglik(scheme, tm, w) for w in worlds])
                # both should be positive (theta is the population maximizer); allow tiny MC noise
                self.assertGreater(dp, -0.02, msg=f"{scheme} param {i} up-perturb: {dp:.4f}")
                self.assertGreater(dm, -0.02, msg=f"{scheme} param {i} down-perturb: {dm:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
