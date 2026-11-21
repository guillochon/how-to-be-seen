"""Turn the cached equivalent-width curves into the paper's numbers.

Reads research/curves.pkl (scripts/run_model.py). Writes research/results.json, which the
figure scripts and the tables in paper/main.tex are taken from, and prints a summary.

    uv run python scripts/results.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, detect, hosts, survey  # noqa: E402
from beacon.const import EV, MEARTH, YR  # noqa: E402

SNRS = (300.0, 1000.0)
LSUN = 3.828e33  # nominal solar luminosity (erg s^-1), IAU 2015 B3
# Hosts below this Teff: UV Tc II results are not reported (metal bound-free opacity omitted).
TC2_MIN_TEFF = 7000.0
NUCLIDE_FOR = {"Tc I": "99Tc", "Tc II": "99Tc", "Pm II": "147Pm"}
ROUTES_FOR = {"Tc I": ("99Tc fission", "99Tc capture", "97Tc capture"),
              "Tc II": ("99Tc fission", "99Tc capture", "97Tc capture"),
              "Pm II": ("147Pm fission", "145Pm capture")}
# Route quoted as the cost of each beacon: neutron capture, keeping the 99Tc lifetime for Tc.
PRIMARY_ROUTE = {"Tc I": "99Tc capture", "Tc II": "99Tc capture", "Pm II": "145Pm capture"}


def luminosity(teff: float) -> float:
    """log L/Lsun of a main-sequence star of this Teff (Mamajek table)."""
    teffs, logl = [], []
    for ln in (ROOT / "research" / "mamajek" / "EEM_dwarf_UBVIJHK_colors_Teff.txt").read_text().splitlines():
        tok = ln.split()
        if len(tok) > 8 and tok[0].endswith("V") and tok[0][0] in "BAFGKM" and tok[1].isdigit():
            try:
                teffs.append(float(tok[1]))
                logl.append(float(tok[4]))
            except ValueError:
                continue
    o = np.argsort(teffs)
    return float(np.interp(teff, np.array(teffs)[o], np.array(logl)[o]))


def main():
    curves = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))
    ground = {i.name: i for i in detect.INSTRUMENTS}
    out = {"hosts": {}, "instruments": [i.__dict__ for i in detect.INSTRUMENTS], "snr": SNRS}
    for h in hosts.host_grid():
        hc = curves[h.key]
        L = 10 ** luminosity(h.teff) * LSUN if not h.solar else LSUN
        rec = dict(label=h.label, teff=h.teff, logg=h.logg, mass=h.mass, mmix=h.mmix, mmix_lo=h.mmix_lo,
                   mmix_hi=h.mmix_hi, MV=survey.absolute_v(h.teff),
                   L_erg_s=L, species={})
        for name, c in hc.items():
            if name == "Tc II" and h.teff < TC2_MIN_TEFF:
                continue
            nuc = NUCLIDE_FOR[name]
            nd = hosts.NUCLIDES[nuc]
            srec = {}
            for inst in detect.INSTRUMENTS:
                for snr in SNRS:
                    le = survey.required_log_eps(c, inst, snr)
                    if not np.isfinite(le):
                        continue
                    m = hosts.isotope_mass(le, nd["A"], h.mmix)
                    e = hosts.fission_energy(m, nuc)
                    bl, _ = survey.best_line(c, inst, snr, le)
                    r = dict(log_eps=le, mass_g=m, mass_lo_g=hosts.isotope_mass(le, nd["A"], h.mmix_lo),
                             mass_hi_g=hosts.isotope_mass(le, nd["A"], h.mmix_hi), mass_earth=m / MEARTH,
                             fission_energy_erg=e, energy_over_L_yr=e / (L * YR), best_line=bl.label)
                    # Upkeep: fission power that replaces what decays (Tc and Pm alike); Sagan type from it.
                    rate = hosts.resupply_rate(m, nuc)  # g/yr
                    p = hosts.fission_energy(rate, nuc) / YR  # erg/s
                    r.update(resupply_g_per_yr=rate, resupply_power_erg_s=p, resupply_power_over_L=p / L,
                             upkeep_W=p / 1e7, sagan_K=(np.log10(p / 1e7) - 6) / 10)
                    # The same beacon made along each production route (other isotopes scale by mass number).
                    routes = {}
                    for route in ROUTES_FOR[name]:
                        rn = hosts.ROUTES[route][0]
                        mr = m * hosts.NUCLIDES[rn]["A"] / nd["A"]
                        pr = hosts.upkeep_power(mr, route) / 1e7
                        routes[route] = dict(mass_g=mr, build_energy_erg=hosts.production_energy(mr, route),
                                             upkeep_W=pr, sagan_K=(np.log10(pr) - 6) / 10)
                    r["routes"] = routes
                    best = PRIMARY_ROUTE[name]
                    r.update(upkeep_best_W=routes[best]["upkeep_W"], sagan_K_best=routes[best]["sagan_K"],
                             best_route=best)
                    srec[f"{inst.name}|{snr:.0f}"] = r
            rec["species"][name] = srec
        out["hosts"][h.key] = rec
    (ROOT / "research" / "results.json").write_text(json.dumps(out, indent=1, default=float))

    key = "8.2 m, R = 140,000"
    for snr in SNRS:
        print(f"\n== {key}, S/N = {snr:.0f}")
        for hk, rec in out["hosts"].items():
            for name, srec in rec["species"].items():
                r = srec.get(f"{key}|{snr:.0f}") or srec.get(f"2.4 m space UV, R = 114,000|{snr:.0f}")
                if r is None:
                    continue
                extra = f"  P/L={r['resupply_power_over_L']:.2e}" if "resupply_power_over_L" in r else ""
                print(f"{hk:3s} {name:6s} log eps={r['log_eps']:6.2f}  M={r['mass_g']:.2e} g "
                      f"({r['mass_earth']:.1e} Mearth)  E/(L yr)={r['energy_over_L_yr']:.2e}  best {r['best_line']}{extra}")
    _ = ground, EV


if __name__ == "__main__":
    main()
