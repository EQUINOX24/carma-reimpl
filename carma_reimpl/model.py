"""CARMA(p, q) model: roots, ACVF, PSD, Kalman likelihood, simulation.

Conventions follow Kelly et al. (2014), arXiv:1402.5978. Time units are arbitrary.
Parameters used here: marginal sd `s` (sqrt of R(0)), AR and MA roots built from
log-quadratic coefficients (Eq. 30). The measurement-error scale nu is fixed to 1 in
the fit (see README).
"""
import numpy as np


def quad_roots(logc):
    """AR/MA roots from log-quadratic coefficients (paper Eq. 30).

    Each pair of coefficients (a, b) gives the quadratic z^2 + b z + a. An odd length
    adds one real root -exp(logc[-1]). Returns `len(logc)` complex roots.
    """
    logc = np.asarray(logc, dtype=float)
    p = len(logc)
    roots = []
    for k in range(p // 2):
        a, b = np.exp(logc[2 * k]), np.exp(logc[2 * k + 1])
        sq = np.sqrt(complex(b * b - 4.0 * a))
        roots += [(-b + sq) / 2.0, (-b - sq) / 2.0]
    if p % 2 == 1:
        roots.append(-np.exp(logc[-1]))
    return np.array(roots, dtype=complex)


def ma_coefs(logc, q):
    """MA coefficients beta_0..beta_q with beta_0 = 1 (paper Sec. 2.1)."""
    if q == 0:
        return np.array([1.0])
    poly = np.poly(quad_roots(logc))  # highest power first, leading coefficient 1
    return (poly[::-1] / poly[-1]).real


def _acvf_coefs(r, beta):
    """Per-root coefficients c_k with R(tau) = sigma^2 * Re(sum_k c_k exp(r_k tau)) (Eq. 4)."""
    p = len(r)
    bpoly = beta[::-1]  # highest power first for polyval
    coefs = np.empty(p, dtype=complex)
    for k in range(p):
        num = np.polyval(bpoly, r[k]) * np.polyval(bpoly, -r[k])
        den = -2.0 * r[k].real
        for l in range(p):
            if l != k:
                den *= (r[l] - r[k]) * (np.conj(r[l]) + r[k])
        coefs[k] = num / den
    return coefs


def unit_variance(r, beta):
    """R(0) for driving-noise sigma = 1."""
    return _acvf_coefs(r, beta).sum().real


def acvf(tau, s, r, beta):
    """Autocovariance R(tau) of the CARMA process with marginal sd s (tau >= 0 array)."""
    sigma2 = s**2 / unit_variance(r, beta)
    c = _acvf_coefs(r, beta)
    tau = np.abs(np.asarray(tau, dtype=float))
    out = np.zeros(tau.shape)
    for ck, rk in zip(c, r):
        out += (ck * np.exp(rk * tau)).real
    return sigma2 * out


def psd(f, s, r, beta):
    """Power spectral density at frequencies f (cycles per time unit), paper Eq. (3)."""
    sigma2 = s**2 / unit_variance(r, beta)
    w = 2j * np.pi * np.asarray(f, dtype=float)
    num = np.abs(np.polyval(beta[::-1], w)) ** 2
    den = np.abs(np.polyval(np.poly(r), w)) ** 2
    return sigma2 * num / den


def loglik(t, y, ysig, s, r, beta, nu=1.0):
    """Gaussian log-likelihood via the Kalman filter (paper Sec. 3.1, Appendix A).

    Assumes y is centered (mean zero). Complex rotated-basis state, O(n p^2).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ysig = np.asarray(ysig, float)
    p = len(r)
    q = len(beta) - 1
    sigma2 = s**2 / unit_variance(r, beta)

    b = np.zeros(p, dtype=complex)
    b[: q + 1] = beta
    U = np.vander(r, p, increasing=True).T  # U[l, k] = r_k**l
    e = np.zeros(p, dtype=complex)
    e[-1] = 1.0
    J = np.linalg.solve(U, e)
    bt = b @ U  # rotated MA row vector
    V = -sigma2 * np.outer(J, np.conj(J)) / (r[:, None] + np.conj(r)[None, :])

    x = np.zeros(p, dtype=complex)
    P = V.copy()
    ll = -0.5 * len(y) * np.log(2 * np.pi)
    dt = np.diff(t)
    for i in range(len(y)):
        if i > 0:
            lam = np.exp(r * dt[i - 1])
            x = lam * x
            P = lam[:, None] * (P - V) * np.conj(lam)[None, :] + V
        m = (bt @ x).real
        v = (bt @ P @ np.conj(bt)).real + nu * ysig[i] ** 2
        K = P @ np.conj(bt) / v
        innov = y[i] - m
        x = x + innov * K
        P = P - v * np.outer(K, np.conj(K))
        ll += -0.5 * (np.log(v) + innov**2 / v)
    return ll


def _cov(t, s, r, beta, ysig, nu=1.0):
    t = np.asarray(t, float)
    return acvf(np.abs(t[:, None] - t[None, :]), s, r, beta) + np.diag(nu * np.asarray(ysig) ** 2)


def loglik_dense(t, y, ysig, s, r, beta, nu=1.0):
    """Dense O(n^3) reference log-likelihood, for tests."""
    S = _cov(t, s, r, beta, ysig, nu)
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, np.asarray(y, float))
    n = len(y)
    return -0.5 * n * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * z @ z


def simulate(t, s, r, beta, ysig, rng, nu=1.0):
    """Exact simulation at times t via dense Cholesky (small n only)."""
    L = np.linalg.cholesky(_cov(t, s, r, beta, ysig, nu))
    return L @ rng.standard_normal(len(t))
