"""Simulate an irregularly sampled CARMA(3,1) light curve, fit CARMA(p,q) by ML,
choose the order by AICc, and write docs/demo_fit.svg.

Run from the package root:  python examples/demo.py
"""
import os

import numpy as np

from carma_reimpl import fit_grid, psd, save_svg, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "demo_fit.svg")


def main():
    rng = np.random.default_rng(42)
    n = 200
    t = np.sort(rng.uniform(0.0, 400.0, n))
    truth_r = np.array([-0.1 + 1.26j, -0.1 - 1.26j, -0.05 + 0j])
    truth_beta = np.array([1.0, 0.5])
    truth_s = 1.0
    ysig = 0.15 * np.ones(n)
    y = simulate(t, truth_s, truth_r, truth_beta, ysig, rng)

    results = fit_grid(t, y, ysig, pmax=3, nstarts=8, seed=1)
    print(f"{'p':>2} {'q':>2} {'NLL':>10} {'AICc':>10} {'starts ok':>9}")
    for r in results:
        print(f"{r.p:>2} {r.q:>2} {r.nll:>10.2f} {r.aicc:>10.2f} {r.nstarts_ok:>9}")
    best = min(results, key=lambda r: r.aicc)
    print(f"Selected CARMA({best.p},{best.q}) by AICc; s = {best.s:.3f}")

    f = np.logspace(np.log10(1 / 400.0), np.log10(1 / (2 * np.min(np.diff(t)))), 300)
    noise = np.mean(ysig**2) * np.ones_like(f)
    panels = [
        dict(title="Simulated irregular light curve", xlabel="time", ylabel="y",
             series=[dict(x=t, y=y, label="data", style="points")]),
        dict(title=f"Power spectrum: truth vs CARMA({best.p},{best.q}) ML fit", xlabel="frequency (cycles / time)",
             ylabel="PSD (latent process)", xlog=True, ylog=True,
             series=[
                 dict(x=f, y=psd(f, truth_s, truth_r, truth_beta), label="truth CARMA(3,1)", style="dashed"),
                 dict(x=f, y=psd(f, best.s, best.r, best.beta), label=f"ML CARMA({best.p},{best.q})"),
                 dict(x=f, y=noise, label="measurement noise level", style="line", color="#999999"),
             ]),
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_svg(OUT, panels)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
