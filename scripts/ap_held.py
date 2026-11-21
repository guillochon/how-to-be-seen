"""A Tc beacon held by the field of a magnetic Ap/Bp star.

A horizontal field slows the vertical drift of an ion where the ion circles the field lines faster than it
collides with the gas (scripts/tc_drift.py, b_trap). For a surface field B, Tc placed above the depth
tau_hold at which b_trap = B stays there, and Tc placed deeper rises to it. For each host temperature and
field strength this script finds tau_hold, then the column of Tc held above tau_hold whose strongest Tc II
ultraviolet lines are as strong as those of the uniform threshold layer above tau5000 = 3 (the threshold
of the 2.4 m space echelle at S/N = 300, research/levitation.json). The inventory and upkeep follow as for
the quiet late-B host, with decay as the only loss.

Matching the summed equivalent width of the six strongest Tc II lines stands in for the full detection
threshold, and Zeeman broadening is ignored.

Geometry of a dipole: the field dips at angle I below the horizontal, tan I = 2 tan(latitude). Radiation
pushes Tc along each field line toward its apex at the magnetic equator, at r = R / cos^2(latitude). Field
lines whose apex lies inside the held layer, of height h, keep their Tc in the atmosphere: latitudes below
sqrt(h / R). Tc on all other lines collects at their apexes in the magnetosphere. Inside that band the
field dips at I ~ 2 latitude, so Tc slides along it to the equator in about R / (2 v), with v the drift
speed halfway up the held layer (slide_yr).

    uv run python scripts/ap_held.py     # writes research/ap_held.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_drift as TD  # noqa: E402
from beacon import ROOT, atlas, atoms, hosts  # noqa: E402
from beacon.synth import Synth  # noqa: E402

HOSTS = ((8000, 1.8), (9000, 2.3), (12250, 3.4))  # Teff, mass (Msun) as for the Am, A0 and late-B hosts
LOGG = 4.0
FIELDS = (300.0, 1000.0, 3000.0, 10000.0)  # G
NLINES = 6


def ew_sum(s, lines, prof):
    return sum(s.ew(ln, prof) for ln in lines)


def main():
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    out = {}
    for teff, mass in HOSTS:
        rec = lev[f"tc|{teff}|+0"]
        le0 = lev[f"threshold|{teff}"]
        lt, T, _rho, z, v, D = TD.drift(teff, LOGG, rec, 10.0)
        b_trap = np.maximum(TD.K * T * TD.C_LIGHT / (TD.E * D[1]), TD.K * T * TD.C_LIGHT / (2 * TD.E * D[2]))
        s = Synth(atlas.model(teff, LOGG), vturb=1.0)
        t5, rx = s.tau_cont_500(), s.m.rhox
        lines = sorted(atoms.kurucz_lines("gf4301.pos", 43, 1, 1900.0, 3300.0), key=lambda ln: -ln.loggf)[:NLINES]
        col3 = float(np.interp(np.log10(TD.TAU_BASE), np.log10(t5), rx))
        target = ew_sum(s, lines, le0)
        r = np.sqrt(TD.G_CGS * mass * TD.MSUN / 10**LOGG)
        area = 4 * np.pi * r**2
        m_uniform = hosts.isotope_mass(le0, TD.A_TC, area * col3 / TD.MSUN)
        rows = {}
        for B in FIELDS:
            # b_trap rises with depth: the held layer is everything above the depth where it reaches B
            lt_hold = float(np.interp(np.log10(B), np.log10(b_trap), lt))
            col_hold = float(np.interp(lt_hold, np.log10(t5), rx))
            held = t5 < 10**lt_hold

            def ew_at(log_scale):
                le = le0 + np.log10(col3 / col_hold) + log_scale  # log_scale = log of column relative to uniform
                return ew_sum(s, lines, np.where(held, le, -20.0))

            lo, hi = -3.0, 3.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ew_at(mid) < target else (lo, mid)
            log_scale = 0.5 * (lo + hi)
            m = m_uniform * 10**log_scale
            p = hosts.upkeep_power(m, "99Tc capture") / 1e7
            h = float(np.interp(lt_hold, lt, z))  # thickness of the held layer, from its base to the model top
            rows[f"{B:g}"] = dict(log_tau_hold=lt_hold, log_eps_in_layer=float(le0 + np.log10(col3 / col_hold) + log_scale),
                                  column_ratio=10**log_scale, mass_kg=m / 1e3, upkeep_W=p, sagan_K=(np.log10(p) - 6) / 10,
                                  layer_height_km=h / 1e5, band_deg=float(np.degrees(np.sqrt(h / r))),
                                  slide_yr=float(r / (2 * np.interp(0.5 * (lt_hold + lt[0]), lt, v)) / TD.YR))
        p0 = hosts.upkeep_power(m_uniform, "99Tc capture") / 1e7
        out[str(teff)] = dict(mass=mass, log_eps_threshold=le0, uniform_mass_kg=m_uniform / 1e3,
                              uniform_K=(np.log10(p0) - 6) / 10,
                              b_trap=[dict(log_tau=float(a), B_G=float(b)) for a, b in zip(lt[::8], b_trap[::8], strict=True)],
                              fields=rows)
        print(f"{teff} K: uniform layer above tau 3: {m_uniform / 1e3:.2e} kg, K = {out[str(teff)]['uniform_K']:.2f}")
        for B, x in rows.items():
            print(f"  B = {B:>6} G: held above log tau {x['log_tau_hold']:5.2f} ({x['layer_height_km']:.0f} km thick), "
                  f"column x{x['column_ratio']:.2f}, {x['mass_kg']:.2e} kg, K = {x['sagan_K']:.2f}, "
                  f"band |lat| < {x['band_deg']:.1f} deg, slides to the equator in {x['slide_yr']:.0f} yr")
    (ROOT / "research" / "ap_held.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
