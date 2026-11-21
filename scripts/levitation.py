"""Radiative levitation of Tc in A and late-B atmospheres, with line-by-line window transfer.

Writes research/levitation.json (appended as each case finishes).

1. Validation at 12,250 K, log g 4: Mg and Fe at solar abundance, Mn at solar and +2 dex,
   to compare with Alecian & Stift (2019): Mg weakly supported, Fe marginal, Mn levitated.
2. Tc from 7000 to 14,000 K at its detection threshold (Tc II, 2.4 m space UV, S/N 300) and
   at 100 times that, compared with gravity.

    uv run python scripts/levitation.py [validate|tc|all] [TEFF ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, detect, hosts, survey  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import AMU, C  # noqa: E402

OUT = ROOT / "research" / "levitation.json"
LOGG = 4.0
VTURB = 0.5
TAUS = (-4, -3, -2, -1, 0, 0.5)
SOLAR = {12: 7.60, 25: 5.43, 26: 7.50}  # Asplund et al. (2009)
TEFFS = (7000, 8000, 9000, 10000, 11000, 12250, 14000)
INST = "2.4 m space UV, R = 114,000"


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(key, val):
    d = load()
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def lineset(z, ions):
    parts = [L._gfall(z, i) for i in ions]
    return L.LineSet(z, *(np.concatenate([getattr(p, f) for p in parts]) for f in ("ion", "wl", "loggf", "elow")),
                     np.vstack([p.gamma for p in parts]))


def unsaturated(atm, el, mass, log_eps):
    """Element acceleration with the continuum flux and no saturation (the earlier estimate)."""
    grid = np.geomspace(880.0, 20500.0, 400)
    Fc = L.continuum_flux(atm, grid)
    k = atm.line_strength(el, log_eps)
    nX = atm.nH * 10.0 ** (log_eps - 12.0)
    g = np.zeros((3, len(atm.T)))
    for ion in range(3):
        sel = el.ion == ion
        if not sel.any():
            continue
        Fl = np.stack([np.interp(np.log(el.wl[sel]), np.log(grid), Fc[d]) for d in range(len(atm.T))], axis=1)
        mom = (k[sel] * Fl * (el.wl[sel, None] * 1e-8) ** 2 / C).sum(axis=0) / C
        g[ion] = mom / np.maximum(nX * atm.frac(el.z)[ion] * mass * AMU, 1e-300)
    return g


def at_taus(atm, arr):
    lt = np.log10(atm.tau5)
    return [float(np.interp(x, lt, np.log10(np.maximum(arr, 1e-30)))) for x in TAUS]


def run_case(key, atm, el, mass, log_eps, nb, corr):
    t0 = time.time()
    g_ion, n = L.grad_element(atm, el, mass, log_eps, nb, corr)
    plain, dw = L.element_grad(atm, g_ion, el.z)
    gu = unsaturated(atm, el, mass, log_eps)
    _, dwu = L.element_grad(atm, gu, el.z)
    rec = dict(log_eps=log_eps, nlines=n, taus=TAUS, log_grad=at_taus(atm, plain), log_grad_diffw=at_taus(atm, dw),
               log_grad_ion=[at_taus(atm, g_ion[i]) for i in range(3)], log_grad_unsat_diffw=at_taus(atm, dwu),
               frac_tau1=[float(np.interp(0.0, np.log10(atm.tau5), atm.frac(el.z)[i])) for i in range(3)],
               seconds=time.time() - t0)
    save(key, rec)
    print(f"{key:28s} log eps={log_eps:5.2f} lines={n:5d} log g_rad(diff-w) " +
          " ".join(f"{v:5.2f}" for v in rec["log_grad_diffw"]) + "  unsat " +
          " ".join(f"{v:5.2f}" for v in rec["log_grad_unsat_diffw"]) + f"  ({rec['seconds']:.0f}s)", flush=True)


def validate():
    atm = L.Atmosphere(12250, LOGG, VTURB)
    corr = L.flux_correction(atm, 12250, LOGG)
    for z, mass, les in ((12, 24.305, (7.60,)), (26, 55.845, (7.50,)), (25, 54.938, (5.43, 7.43))):
        nb = L.neighbour_lines(atm, exclude_z=z)
        el = lineset(z, (0, 1, 2) if z != 12 else (0, 1))
        for le in les:
            run_case(f"validate|{z}|{le:.2f}", atm, el, mass, le, nb, corr)


def threshold(teff):
    mass = 2.5
    h = hosts.Host("x", "x", teff, LOGG, mass, mass * 10**-5.5, mass * 1e-10, mass * 1e-5, 1.0, 10.0, 0.0)
    c = survey.build_curves(h)
    return survey.required_log_eps(c["Tc II"], {i.name: i for i in detect.INSTRUMENTS}[INST], 300.0)


def tc(teffs=TEFFS):
    el = L.tc_lines()
    for teff in teffs:
        le0 = threshold(teff)
        save(f"threshold|{teff}", le0)
        atm = L.Atmosphere(teff, LOGG, VTURB)
        corr = L.flux_correction(atm, teff, LOGG)
        nb = L.neighbour_lines(atm, exclude_z=43)
        for dle in (0.0, 2.0):
            run_case(f"tc|{teff}|{dle:+.0f}", atm, el, 98.906, le0 + dle, nb, corr)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("validate", "all"):
        validate()
    if what in ("tc", "all"):
        tc(tuple(int(t) for t in sys.argv[2:]) or TEFFS)
