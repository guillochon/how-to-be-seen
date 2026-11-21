"""Cost of a Tc II beacon in a quiet, slowly rotating late-B (HgMn-like) star that mixes nothing below
its atmosphere, where radiation holds the Tc (research/levitation.json).

The star: 12,250 K, log g 4 and 3.4 Msun, the mass that Alecian & Stift (2019) derive for their
12,000 K, log g 4 model. The mixed layer is the atmosphere down to tau5000 = 3, the deepest level of the
levitation calculation. Tc is lost by decay and carried off by a wind at the lowest rate that fits Mg
and Fe in their models, 1.7e-14 Msun/yr (and with no wind).

    uv run python scripts/quiet_host.py     # writes research/quiet_host.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, hosts  # noqa: E402
from beacon import levitation as L  # noqa: E402

TEFF, LOGG, MSTAR = 12250, 4.0, 3.4
TAU_BASE = 3.0
MDOT = 1.7e-14  # Msun/yr
G, MSUN = 6.674e-8, 1.989e33


def main():
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    le = lev[f"threshold|{TEFF}"]
    atm = L.Atmosphere(TEFF, LOGG, 0.5)
    col = float(np.interp(np.log10(TAU_BASE), np.log10(atm.tau5), atm.rhox))  # g cm^-2
    r = np.sqrt(G * MSTAR * MSUN / 10**LOGG)
    mmix = 4 * np.pi * r**2 * col / MSUN
    t_half = hosts.NUCLIDES["99Tc"]["t_half"]
    out = dict(teff=TEFF, logg=LOGG, mstar=MSTAR, log_eps=le, column_g_cm2=col, mmix_msun=mmix,
               log_mmix_frac=float(np.log10(mmix / MSTAR)), mdot_msun_yr=MDOT, cases={})
    for name, mdot in (("no wind", 0.0), ("wind", MDOT)):
        m = hosts.isotope_mass(le, 98.906, mmix)
        p = hosts.upkeep_power(m, "99Tc capture") / 1e7 * (1 + mdot / mmix * t_half / np.log(2))
        out["cases"][name] = dict(mass_kg=m / 1e3, upkeep_W=p, sagan_K=(np.log10(p) - 6) / 10,
                                  wind_time_yr=mmix / mdot if mdot else None)
    (ROOT / "research" / "quiet_host.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
