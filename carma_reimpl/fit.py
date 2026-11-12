"""Multi-start maximum likelihood for CARMA(p, q) and AICc order selection.

Parameter vector theta = [log s, log-quadratic AR block (p), log-quadratic MA block (q)].
The mean is fixed to the sample mean and nu is fixed to 1 (v1 simplification).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .model import loglik, ma_coefs, quad_roots

BIG = 1e10


@dataclass
class FitResult:
    p: int
    q: int
    s: float
    r: np.ndarray
    beta: np.ndarray
    mu: float
    nll: float
    aicc: float
    theta: np.ndarray = field(repr=False)
    nstarts_ok: int = 0


def _nll(theta, t, y, ysig, p, q):
    if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 40:
        return BIG  # keeps exp() finite; far outside any plausible scale
    s = np.exp(theta[0])
    with np.errstate(all="ignore"):
        r = quad_roots(theta[1 : 1 + p])
        beta = ma_coefs(theta[1 + p : 1 + p + q], q)
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(beta)):
        return BIG
    if p > 1:
        gap = np.abs(r[:, None] - r[None, :]) + np.eye(p) * 1e300
        if np.min(gap) < 1e-8 * max(1.0, np.max(np.abs(r))):
            return BIG  # repeated roots: outside the model (paper Sec. 3.1)
    with np.errstate(all="ignore"):
        try:
            val = -loglik(t, y, ysig, s, r, beta)
        except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError, OverflowError):
            return BIG
    return float(val) if np.isfinite(val) else BIG


def _random_start(rng, t, p, q, sd):
    T = t[-1] - t[0]
    dtmin = np.min(np.diff(t))
    lo = np.log(2 * np.pi / T) - 1.0  # log angular frequency range, padded
    hi = np.log(2 * np.pi / dtmin) + 1.0
    # quadratic coefficients are products/sums of roots, so use a wider range for "a"
    def block(m):
        out = []
        for _ in range(m // 2):
            out += [rng.uniform(2 * lo, 2 * hi), rng.uniform(lo, hi)]
        if m % 2:
            out.append(rng.uniform(lo, hi))
        return out

    return np.array([np.log(sd)] + block(p) + block(q))


def fit(t, y, ysig, p, q, nstarts=8, seed=0):
    """Multi-start L-BFGS-B fit of CARMA(p, q). Returns the best FitResult."""
    t = np.asarray(t, float)
    ysig = np.asarray(ysig, float)
    mu = float(np.mean(y))
    yc = np.asarray(y, float) - mu
    n = len(yc)
    rng = np.random.default_rng(seed)
    sd = float(np.std(yc))
    best, ok = None, 0
    for _ in range(nstarts):
        x0 = _random_start(rng, t, p, q, sd)
        res = minimize(_nll, x0, args=(t, yc, ysig, p, q), method="L-BFGS-B")
        if res.fun < BIG / 2:
            ok += 1
            if best is None or res.fun < best.fun:
                best = res
    k = 1 + p + q  # free parameters: s + AR + MA (mu and nu fixed)
    if best is None or n - k - 1 <= 0:
        return FitResult(p, q, np.nan, np.full(p, np.nan + 0j), np.full(q + 1, np.nan), mu,
                         np.inf, np.inf, np.full(1 + p + q, np.nan), ok)
    nll = float(best.fun)
    aicc = 2 * k + 2 * nll + 2 * k * (k + 1) / (n - k - 1)
    theta = best.x
    return FitResult(p, q, float(np.exp(theta[0])), quad_roots(theta[1 : 1 + p]),
                     ma_coefs(theta[1 + p :], q), mu, nll, float(aicc), theta, ok)


def fit_grid(t, y, ysig, pmax=3, nstarts=8, seed=0):
    """Fit all (p, q) with 1 <= p <= pmax and 0 <= q < p. Returns a list of FitResult."""
    out = []
    for p in range(1, pmax + 1):
        for q in range(p):
            out.append(fit(t, y, ysig, p, q, nstarts=nstarts, seed=seed + 10 * p + q))
    return out
