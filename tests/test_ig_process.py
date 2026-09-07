import unittest

import numpy as np

import src.ig_process as ig


class TestIGBasics(unittest.TestCase):
    def setUp(self):
        self.theta = (-1.88, 1.73, 0.653, 0.449)  # connector parameters (Ma et al. 2021)
        self.omega = 30.0

    def test_log_pdf_against_scipy(self):
        from scipy.stats import invgauss

        for a, b in [(2.0, 5.0), (10.3, 600.0), (0.5, 1.2)]:
            for y in [0.5, 2.0, 5.0, 20.0]:
                # scipy invgauss uses mu, scale with shape b = 1/scale... Var = mu * scale^2? no:
                # scipy: invgauss(mu=mu, scale=s): mean = mu*s, shape param of IG(mu*s, s)... we
                # compare via direct density formula instead, plus MC mass check below.
                self.assertAlmostEqual(
                    float(np.exp(ig.log_ig_pdf(y, a, b))),
                    np.sqrt(b / (2 * np.pi * y**3)) * np.exp(-b * (y - a) ** 2 / (2 * a**2 * y)),
                    places=12,
                )

    def test_pdf_integrates_to_one(self):
        for a, b in [(2.0, 5.0), (10.3, 600.0), (35.0, 1400.0)]:
            ys = np.linspace(1e-6, a * 40 + 50 * a / b, 400_001)
            dens = np.exp(ig.log_ig_pdf(ys, a, b))
            integral = np.trapezoid(dens, ys)
            self.assertAlmostEqual(integral, 1.0, places=5)

    def test_survival_consistent_with_pdf(self):
        # P(Y>y) from ig_survival == integral of pdf above y
        a, b = 10.3, 600.0
        for y in [5.0, 10.0, 15.0, 25.0]:
            ys = np.linspace(y, 200.0, 200_001)
            integral = np.trapezoid(np.exp(ig.log_ig_pdf(ys, a, b)), ys)
            self.assertAlmostEqual(float(ig.ig_survival(y, a, b)), integral, places=5)

    def test_moments_of_sampler(self):
        rng = np.random.default_rng(7)
        a, b = 10.0, 100.0
        draws = np.array([rng.wald(a, b) for _ in range(200_000)])
        self.assertAlmostEqual(draws.mean(), a, delta=0.05)
        self.assertAlmostEqual(draws.var(), a**3 / b, delta=0.2)

    def test_passage_cdf_matches_montecarlo(self):
        rng = np.random.default_rng(11)
        theta, omega, x = self.theta, self.omega, 1.0
        a0, a1, lam, q = theta
        mu = np.exp(a0 + a1 * x)
        for t in [200.0, 1000.0, 3000.0]:
            # simulate paths at fine grid and check crossing; increments over grid are exact
            m = 60_000
            steps = 400
            grid = np.linspace(0.0, t, steps + 1)[1:]
            dlam = np.diff(np.concatenate([[0.0], grid**q]))
            inc_mean = mu * dlam
            inc_shape = lam * dlam**2
            lev = np.zeros(m)
            crossed = np.zeros(m, dtype=bool)
            for j in range(steps):
                lev = lev + rng.wald(inc_mean[j], inc_shape[j], size=m)
                crossed |= lev >= omega
                if crossed.all():
                    break
            mc = crossed.mean()
            an = float(ig.passage_cdf(t, x, theta, omega))
            self.assertAlmostEqual(mc, an, delta=3.5 * np.sqrt(an * (1 - an) / m) + 1e-3)

    def test_passage_quantile_roundtrip(self):
        for p in [0.05, 0.1, 0.5, 0.9]:
            t = float(ig.passage_quantile(p, 1.0, self.theta, self.omega)[0])
            back = float(ig.passage_cdf(t, 1.0, self.theta, self.omega))
            self.assertAlmostEqual(back, p, places=4)

    def test_xi_approx_close_to_exact(self):
        for x in [0.0, 0.46, 1.0]:
            ex = float(ig.passage_quantile(0.1, x, self.theta, self.omega)[0])
            ap = float(ig.xi_p_approx(0.1, x, self.theta, self.omega))
            self.assertAlmostEqual(np.log(ap), np.log(ex), delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
