"""A photospheric beacon of 226Ra, read in the Ra II lines, in each host of the paper.

Same hosts, instruments and detection rule (Z = 5) as the Tc and Pm beacons (scripts/run_model.py,
scripts/results.py); the lines are the Ra II lines with NIST transition probabilities
(beacon/data/ra2_nist.csv), and the Ra I partition function comes from the NIST levels
(beacon/data/ra1_levels.csv). 226Ra has no route from a stable target. Its supply is the 238U chain: a stock of
natural uranium in secular equilibrium makes 226Ra as fast as the radium it holds decays, so a beacon of mass M
is kept up by a uranium stock of M times hosts.uranium_per_radium(), with no energy spent beyond chemistry.
That stock is compared with the uranium a Tc beacon bred from 98Mo burns: one spare fission neutron per Tc atom,
1 / (nu_bar - 1) fissions of 235U, and natural uranium holds 1 - U238_FRACTION of 235U.

In the Sun the strongest line, Ra II 3814, lies in a deep blend. As scripts/blending.py does for Tc I and Pm II,
the script also gives the solar threshold from the clean lines alone: those around which the Kurucz (2005) solar
flux atlas stays above 0.9 of the continuum within +/- 0.05 A.

Writes research/radium_beacon.json and prints the thresholds.

    uv run python scripts/radium_beacon.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ, ROOT, atoms, detect, hosts, survey  # noqa: E402
from beacon.const import AMU, YR  # noqa: E402

SNR = 300.0
CLEAN, HALF = 0.9, 0.05  # as in scripts/blending.py
TC_KEY = {"Tc I": "8.2 m, R = 140,000|300", "Tc II": "2.4 m space UV, R = 114,000|300"}


def natural_u_burned_per_g_tc() -> float:
    """Grams of natural uranium whose 235U fissions supply the spare neutrons for 1 g of 99Tc bred from 98Mo."""
    fissions_per_atom = 1.0 / (hosts.NU_BAR - 1.0)
    g_u235 = fissions_per_atom * 235.0439 / hosts.NUCLIDES["99Tc"]["A"]
    return g_u235 / (1.0 - hosts.U238_FRACTION)


def main():
    survey.SPECIES = {"Ra II": lambda: atoms.ra2_lines()}
    tc = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    u_per_ra = hosts.uranium_per_radium()
    out = dict(snr=SNR, uranium_per_radium=u_per_ra, natural_u_per_g_tc=natural_u_burned_per_g_tc(), hosts={})
    for h in hosts.host_grid():
        c = survey.build_curves(h)["Ra II"]
        rec = dict(label=h.label, teff=h.teff, mmix=h.mmix, instruments={})
        for inst in detect.INSTRUMENTS:
            le = survey.required_log_eps(c, inst, SNR)
            if not np.isfinite(le):
                continue
            m = hosts.isotope_mass(le, hosts.NUCLIDES["226Ra"]["A"], h.mmix)  # g
            bl, _ = survey.best_line(c, inst, SNR, le)
            rec["instruments"][inst.name] = dict(log_eps=le, mass_kg=m / 1e3, uranium_stock_kg=m * u_per_ra / 1e3,
                                                 decay_kg_per_yr=hosts.resupply_rate(m, "226Ra") / 1e3,
                                                 best_line=bl.label)
        best = min(rec["instruments"].values(), key=lambda r: r["mass_kg"], default=None)
        rec["best"] = best
        # the cheapest Tc beacon of the same host at the same S/N, for comparison
        tcs = [(sp, tc[h.key]["species"][sp][k]) for sp, k in TC_KEY.items() if k in tc[h.key]["species"].get(sp, {})]
        if tcs:
            sp, r = min(tcs, key=lambda q: q[1]["mass_g"])
            rate = hosts.resupply_rate(r["mass_g"], "99Tc")  # g/yr
            rec["tc"] = dict(species=sp, log_eps=r["log_eps"], mass_kg=r["mass_g"] / 1e3, K=r["sagan_K_best"],
                             natural_u_burned_kg_per_yr=rate * out["natural_u_per_g_tc"] / 1e3)
        if h.solar:
            raw = np.loadtxt(KURUCZ / "sun" / "solarfluxintwl.asc", skiprows=14)
            wl, fl = 10.0 * raw[:, 0], raw[:, 1]
            fmin = [float(fl[np.abs(wl - ln.wl) < HALF].min()) if (np.abs(wl - ln.wl) < HALF).any() else np.nan
                    for ln in c.lines]
            keep = [i for i, f in enumerate(fmin) if f > CLEAN]
            cc = copy.copy(c)
            cc.lines = [c.lines[i] for i in keep]
            cc.ew, cc.fwhm_int, cc.flux_ratio = c.ew[keep], c.fwhm_int[keep], c.flux_ratio[keep]
            inst = {i.name: i for i in detect.INSTRUMENTS}["8.2 m, R = 140,000"]
            le = survey.required_log_eps(cc, inst, SNR)
            m = hosts.isotope_mass(le, hosts.NUCLIDES["226Ra"]["A"], h.mmix)
            bl, _ = survey.best_line(cc, inst, SNR, le)
            rec["clean"] = dict(lines=[c.lines[i].label for i in keep], atlas_min={ln.label: f for ln, f in zip(c.lines, fmin, strict=True)},
                                log_eps=le, mass_kg=m / 1e3, uranium_stock_kg=m * u_per_ra / 1e3, best_line=bl.label)
            print(f"    Sun, clean Ra II lines only ({len(keep)} of {len(c.lines)}): log eps {le:+.2f}, M_Ra {m / 1e3:.1e} kg, "
                  f"U stock {m * u_per_ra / 1e3:.1e} kg ({bl.label})", flush=True)
        out["hosts"][h.key] = rec
        if best:
            line = f"{h.key:3s} {h.teff:6.0f} K  Ra II log eps {best['log_eps']:+.2f}  M_Ra {best['mass_kg']:.1e} kg  " \
                   f"U stock {best['uranium_stock_kg']:.1e} kg  ({best['best_line']})"
            if "tc" in rec:
                line += f"  | {rec['tc']['species']} M_Tc {rec['tc']['mass_kg']:.1e} kg, U burned {rec['tc']['natural_u_burned_kg_per_yr']:.1e} kg/yr"
            print(line, flush=True)
    (ROOT / "research" / "radium_beacon.json").write_text(json.dumps(out, indent=1, default=float))
    _ = AMU, YR


if __name__ == "__main__":
    main()
