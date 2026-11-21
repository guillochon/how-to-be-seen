"""Flux density of a V = 0 star at 5500 A, from the solar model rather than a quoted calibration.

f_lambda(V=0) = F_cont(5500) * <atlas residual flux, 5400-5600 A> * (R_sun / 10 pc)^2 * 10^(0.4 M_V,sun),
with F_cont from the ATLAS9 solar model and this paper's continuum opacity, the line blanketing from
the Kurucz (2005) flux atlas, and M_V,sun from the Mamajek dwarf table (G2V). beacon/detect.py uses
the printed value.

    uv run python scripts/flux_zero_point.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ, atlas, atoms, survey, synth  # noqa: E402
from beacon.const import PC, RSUN  # noqa: E402

s = synth.Synth(atlas.sun())
_, _, F = s.profile(atoms.Line(5500.0, -9.0, 0.0, 26, 0), -20.0, npts=3)
raw = np.loadtxt(KURUCZ / "sun" / "solarfluxintwl.asc", skiprows=14)
w, f = 10 * raw[:, 0], raw[:, 1]
blanket = f[(w > 5400) & (w < 5600)].mean()
mv_sun = survey.absolute_v(5777.0)
fv0 = F * 1e-8 * blanket * (RSUN / (10 * PC)) ** 2 * 10 ** (0.4 * mv_sun)
print(f"blanketing {blanket:.4f}, M_V,sun {mv_sun:.2f}, f_lambda(V=0) = {fv0:.3e} erg s^-1 cm^-2 A^-1")
