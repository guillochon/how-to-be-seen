"""Self-shielding of a magnetospheric Tc II disk (scripts/disk_beacon.py): how much its own lines cut the
radiative push on the ions at its mid-plane.

Each Tc II line (Kurucz gf4301.pos, 900-20,000 A) pushes in proportion to gf, its lower-level population at
T_exc = 3000 K and the stellar flux at its wavelength (Castelli & Kurucz). Inside a disk of surface density
Sigma, the starlight reaching the mid-plane has crossed a column Sigma / 2, so each line's push per ion falls by
its Doppler curve-of-growth factor W(tau0) / (tau0 W'(0)), with b = 1.5 km/s. The disk holds while the reduced
push still exceeds gravity, that is while the reduction stays above g / g_rad.

    uv run python scripts/disk_shield.py     # writes research/disk_shield.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, atoms  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import LINE_CROSS  # noqa: E402

T_EXC = 3000.0
B = 1.5e5
SIGMAS = (1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18)
TEFFS = (8000, 9000, 12250)


def reduction(teff, sigma, lines, U):
    w, fnu, _ = L.castelli_flux(teff, 4.0)
    x = np.linspace(-6, 6, 241)
    num = den = 0.0
    for ln in lines:
        pop = np.exp(-atoms.HC_K * ln.elow / T_EXC) / U
        k = 10**ln.loggf * pop * np.interp(ln.wl, w, fnu)
        tau0 = LINE_CROSS * 10**ln.loggf * sigma / 2 * pop * ln.wl * 1e-8 / (np.sqrt(np.pi) * B)
        cog = np.trapezoid(-np.expm1(-tau0 * np.exp(-x * x)), x) / np.sqrt(np.pi) / tau0 if tau0 > 1e-6 else 1.0
        num += k * cog
        den += k
    return num / den


def main():
    lines = atoms.kurucz_lines("gf4301.pos", 43, 1, 900.0, 20000.0)
    U = float(atoms.partition(43, 1, np.array([T_EXC]))[0])
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    out = {}
    for teff in TEFFS:
        g_ratio = 10 ** min(lev[f"tc|{teff}|+0"]["log_grad_diffw"]) / 1e4
        red = {f"{s:g}": reduction(teff, s, lines, U) for s in SIGMAS}
        s_max = max((s for s in SIGMAS if red[f"{s:g}"] * g_ratio > 1), default=None)
        out[str(teff)] = dict(g_rad_over_g_thin=g_ratio, reduction=red, max_sigma_held=s_max)
        print(teff, f"g_rad/g {g_ratio:.0f}", " ".join(f"{k}:{v:.3f}" for k, v in red.items()), f"held up to {s_max:g}")
    (ROOT / "research" / "disk_shield.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
