"""Validate the line-formation code against isolated solar Fe I lines.

Selects Fe I lines with laboratory oscillator strengths (Kurucz reference codes FMW and O),
synthesizes them in the ATLAS9 solar model with log eps(Fe) = 7.50, and compares with
equivalent widths measured in the Kurucz (2005) solar flux atlas. Writes
research/validation_sun.csv; tests/test_validation.py and paper/figures/make_validation.py read it.

    uv run python scripts/validate_sun.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ, ROOT, atlas, atoms, synth

LOG_EPS_FE = 7.50  # Asplund et al. (2009)
ISOLATION = 0.30  # Angstrom: no other atlas feature deeper than 2 per cent within this distance


def load_atlas():
    rows = np.loadtxt(KURUCZ / "sun" / "solarfluxintwl.asc", skiprows=14)
    return 10.0 * rows[:, 0], rows[:, 1]


def measure(wl, fl, w0):
    """Equivalent width (mA) of the feature at w0, or None if it is blended or absent."""
    sel = (wl > w0 - 1.0) & (wl < w0 + 1.0)
    x, y = wl[sel], fl[sel]
    cont = np.percentile(y, 95)
    y = y / cont
    core = np.abs(x - w0) < 0.03
    if not core.any():
        return None
    imin = np.argmin(np.where(core, y, 9.0))
    depth = 1.0 - y[imin]
    if depth < 0.03 or abs(x[imin] - w0) > 0.015:
        return None
    xc = x[imin]
    # half width at half depth
    half = 1.0 - depth / 2
    left = xc - x[(x < xc) & (y > half)].max() if ((x < xc) & (y > half)).any() else None
    right = x[(x > xc) & (y > half)].min() - xc if ((x > xc) & (y > half)).any() else None
    if left is None or right is None or abs(left - right) > 0.3 * (left + right) / 2:
        return None  # asymmetric: blended
    hwhm = 0.5 * (left + right)
    if hwhm > 0.12:
        return None  # sitting in the wing of a strong (e.g. Balmer) line
    win = np.abs(x - xc) < 3.0 * hwhm
    out = (np.abs(x - xc) > 3.0 * hwhm) & (np.abs(x - xc) < ISOLATION)
    if out.any() and y[out].min() < 0.98:
        return None
    ew = 1e3 * np.trapezoid(1.0 - y[win], x[win])
    gauss = 1e3 * 1.064 * depth * 2 * hwhm  # EW of a Gaussian with this depth and width
    if not (0.7 < ew / gauss < 1.4):
        return None  # profile shape inconsistent with a single unblended line
    return ew, depth


def main():
    wl, fl = load_atlas()
    s = synth.Synth(atlas.sun())
    lines = []
    raw = (KURUCZ / "gf2600.all").read_text().splitlines()
    for ln in raw:
        ref = ln[98:102].strip()
        if ref not in ("FMW", "O"):
            continue
        try:
            w = 10 * float(ln[0:11])
        except ValueError:
            continue
        if not (4000 < w < 6800):
            continue
        e1, e2 = abs(float(ln[24:36])), abs(float(ln[52:64]))
        gr, gs, gw = float(ln[80:86]), float(ln[86:92]), float(ln[92:98])
        lines.append(atoms.Line(w, float(ln[11:18]), min(e1, e2), 26, 0,
                                gr or None, gs or None, gw or None))
    out = []
    for line in lines:
        ew_mod = s.ew(line, LOG_EPS_FE)
        if not (5.0 < ew_mod < 90.0):
            continue
        m = measure(wl, fl, line.wl)
        if m is None:
            continue
        out.append((line.wl, line.loggf, line.elow / 8065.544, ew_mod, m[0], m[1]))
    out.sort()
    path = ROOT / "research" / "validation_sun.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wl_A", "loggf", "elow_eV", "ew_model_mA", "ew_atlas_mA", "depth"])
        for r in out:
            w.writerow([f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.3f}"])
    d = np.log10(np.array([r[4] for r in out]) / np.array([r[3] for r in out]))
    blue = np.array([r[0] for r in out]) < 4500
    print(f"{len(out)} isolated lines; log(EW_atlas/EW_model) mean {d.mean():+.3f}, std {d.std():.3f}")
    print(f"  blue (<4500 A): n={blue.sum()} mean {d[blue].mean():+.3f};  red: n={(~blue).sum()} mean {d[~blue].mean():+.3f}")


if __name__ == "__main__":
    main()
