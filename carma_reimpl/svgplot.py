"""Dependency-free SVG line/point plotter (linear or log axes)."""
import math

import numpy as np

_W, _H_PANEL, _TOP = 900, 300, 40
_LEFT, _RIGHT, _PTOP, _PBOTTOM = 90, 30, 35, 55
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#555555"]


def _ticks(lo, hi, log):
    if log:
        return [10.0**k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    return list(np.linspace(lo, hi, 6))


def _fmt(v):
    return f"{v:.3g}"


def _panel(panel, y0):
    xlog, ylog = panel.get("xlog", False), panel.get("ylog", False)
    xs, ys = [], []
    for s in panel["series"]:
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y"], float)
        m = np.isfinite(x) & np.isfinite(y)
        if xlog:
            m &= x > 0
        if ylog:
            m &= y > 0
        s["_m"] = m
        xs.append(x[m])
        ys.append(y[m])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    x0, x1 = float(xs.min()), float(xs.max())
    y0v, y1v = float(ys.min()), float(ys.max())
    if not xlog:
        pad = 0.05 * (x1 - x0 or 1.0)
        x0, x1 = x0 - pad, x1 + pad
    if not ylog:
        pad = 0.08 * (y1v - y0v or 1.0)
        y0v, y1v = y0v - pad, y1v + pad
    if xlog:
        x0, x1 = 10 ** math.floor(math.log10(x0)), 10 ** math.ceil(math.log10(x1))
    if ylog:
        y0v, y1v = 10 ** math.floor(math.log10(y0v)), 10 ** math.ceil(math.log10(y1v))

    L, R = _LEFT, _W - _RIGHT
    T, B = y0 + _PTOP, y0 + _H_PANEL - _PBOTTOM

    def mx(v):
        f = (math.log10(v) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) if xlog else (v - x0) / (x1 - x0)
        return L + f * (R - L)

    def my(v):
        f = (math.log10(v) - math.log10(y0v)) / (math.log10(y1v) - math.log10(y0v)) if ylog else (v - y0v) / (y1v - y0v)
        return B - f * (B - T)

    out = [f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" fill="none" stroke="#333"/>']
    out.append(f'<text x="{(L+R)/2}" y="{T-12}" text-anchor="middle" font-size="15">{panel["title"]}</text>')
    for v in _ticks(x0, x1, xlog):
        out.append(f'<line x1="{mx(v)}" y1="{B}" x2="{mx(v)}" y2="{B+5}" stroke="#333"/>')
        out.append(f'<text x="{mx(v)}" y="{B+18}" text-anchor="middle" font-size="11">{_fmt(v)}</text>')
    for v in _ticks(y0v, y1v, ylog):
        out.append(f'<line x1="{L-5}" y1="{my(v)}" x2="{L}" y2="{my(v)}" stroke="#333"/>')
        out.append(f'<text x="{L-8}" y="{my(v)+4}" text-anchor="end" font-size="11">{_fmt(v)}</text>')
    out.append(f'<text x="{(L+R)/2}" y="{B+40}" text-anchor="middle" font-size="12">{panel["xlabel"]}</text>')
    out.append(f'<text x="18" y="{(T+B)/2}" transform="rotate(-90 18 {(T+B)/2})" text-anchor="middle" font-size="12">{panel["ylabel"]}</text>')

    for i, s in enumerate(panel["series"]):
        color = s.get("color", _COLORS[i % len(_COLORS)])
        x = np.asarray(s["x"], float)[s["_m"]]
        y = np.asarray(s["y"], float)[s["_m"]]
        pts = [(mx(a), my(b)) for a, b in zip(x, y)]
        style = s.get("style", "line")
        if style in ("line", "dashed"):
            dash = ' stroke-dasharray="6,4"' if style == "dashed" else ""
            d = " ".join(f"{a:.2f},{b:.2f}" for a, b in pts)
            out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        if style == "points":
            for a, b in pts:
                out.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.5" fill="{color}"/>')
        ly = T + 16 + 16 * i
        lx = R - 200
        dash = ' stroke-dasharray="6,4"' if style == "dashed" else ""
        if style == "points":
            out.append(f'<circle cx="{lx+12}" cy="{ly-4}" r="4" fill="{color}"/>')
        else:
            out.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+24}" y2="{ly-4}" stroke="{color}" stroke-width="2"{dash}/>')
        out.append(f'<text x="{lx+30}" y="{ly}" font-size="12">{s["label"]}</text>')
    return out


def save_svg(path, panels):
    """Write a vertical stack of panels to an SVG file.

    Each panel: dict(title, xlabel, ylabel, series=[dict(x, y, label, style, color?)],
    xlog=False, ylog=False). style is 'line', 'dashed', or 'points'.
    """
    height = _TOP + _H_PANEL * len(panels) + 10
    body = []
    for i, p in enumerate(panels):
        body += _panel(p, _TOP + i * _H_PANEL)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{height}" '
           f'font-family="sans-serif" background="white">',
           f'<rect width="{_W}" height="{height}" fill="white"/>', *body, "</svg>"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg) + "\n")
