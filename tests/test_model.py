import unittest

import numpy as np

from carma_reimpl import acvf, loglik, loglik_dense, psd, quad_roots, ma_coefs


class TestModel(unittest.TestCase):
    def test_car1_acvf_and_psd(self):
        alpha, s = 0.3, 1.7
        r = quad_roots([np.log(alpha)])
        tau = np.linspace(0, 10, 7)
        np.testing.assert_allclose(acvf(tau, s, r, np.array([1.0])), s**2 * np.exp(-alpha * tau), rtol=1e-10)
        f = np.logspace(-2, 1, 9)
        w = 2 * np.pi * f
        np.testing.assert_allclose(psd(f, s, r, np.array([1.0])), 2 * alpha * s**2 / (alpha**2 + w**2), rtol=1e-10)

    def test_acvf_is_fourier_transform_of_psd(self):
        r = np.array([-0.1 + 1.26j, -0.1 - 1.26j, -0.05 + 0j])
        beta = np.array([1.0, 0.5])
        f = np.linspace(-100, 100, 2_000_001)
        df = f[1] - f[0]
        R0 = np.sum(psd(f, 1.0, r, beta)) * df
        self.assertAlmostEqual(R0, acvf(np.array([0.0]), 1.0, r, beta)[0], delta=1e-3 * R0)

    def test_kalman_matches_dense_carma31(self):
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 50, 40))
        y = rng.standard_normal(40)
        ysig = rng.uniform(0.2, 0.6, 40)
        r = np.array([-0.1 + 1.0j, -0.1 - 1.0j, -0.3 + 0j])
        beta = np.array([1.0, 0.7, 0.2])[:2]
        for nu in (1.0, 1.3):
            self.assertAlmostEqual(loglik(t, y, ysig, 0.8, r, beta, nu=nu),
                                   loglik_dense(t, y, ysig, 0.8, r, beta, nu=nu), places=8)

    def test_kalman_matches_dense_car1(self):
        rng = np.random.default_rng(1)
        t = np.sort(rng.uniform(0, 20, 30))
        y = rng.standard_normal(30)
        ysig = np.full(30, 0.3)
        r = quad_roots([np.log(0.4)])
        self.assertAlmostEqual(loglik(t, y, ysig, 1.1, r, np.array([1.0])),
                               loglik_dense(t, y, ysig, 1.1, r, np.array([1.0])), places=8)

    def test_ma_coefs_normalized(self):
        b = ma_coefs(np.log([0.5, 0.8]), 2)
        self.assertAlmostEqual(b[0], 1.0)
        self.assertEqual(len(b), 3)


if __name__ == "__main__":
    unittest.main()
