"""Radii of the Tc-search stars that have no Gaia DR3 FLAME or published radius, from the Stefan-Boltzmann law.

L follows from V and the parallax in SIMBAD and the bolometric correction of Flower (1996) as a polynomial in
log Teff, with the coefficients of Torres (2010, 2010AJ....140.1158T, Table 1; text in research/torres2010/).
The solar bolometric magnitude comes from the same polynomial, as Torres requires: M_bol,sun = V_sun + 31.572 +
BC_V,sun with V_sun = -26.76. Extinction is neglected. The same is done for the stars with a radius, mostly
from FLAME: the Stefan-Boltzmann radius is lower by a few per cent below 12,000 K and by about 10% above (the
neglected extinction and the bolometric scale), so radius_cal divides it by the median ratio in its Teff band.
Two stars that SIMBAD resolves to one object (the components of a binary) get no radius: their V is combined.
Writes research/tc_limits/radii.csv, whose radius_cal scripts/tc_limits.py reads for every star whose radius is
otherwise unknown.

    uv run python scripts/radii.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402
from archive_coverage import SIMBAD, tap  # noqa: E402

OUT = T.OUT / "radii.csv"
TEFF_SUN = 5772.0  # K, IAU 2015 Resolution B3 nominal value
V_SUN = -26.76  # Torres (2010)
# V for stars that have none in SIMBAD: (V, source)
V_EXTRA = {"hd86986": (7.990, "Mermilliod UBV means, VizieR II/168/ubvmeans")}
TEFF_BANDS = ((0, 9000), (9000, 12000), (12000, 30000))  # K, for the calibration to the adopted radii
# Flower (1996) BC_V = sum c_i (log Teff)^i, from Torres (2010) Table 1, for log Teff < 3.70, 3.70-3.90, > 3.90
BC_COEF = (
    (-0.190537291496456e05, 0.155144866764412e05, -0.421278819301717e04, 0.381476328422343e03),
    (-0.370510203809015e05, 0.385672629965804e05, -0.150651486316025e05, 0.261724637119416e04,
     -0.170623810323864e03),
    (-0.118115450538963e06, 0.137145973583929e06, -0.636233812100225e05, 0.147412923562646e05,
     -0.170587278406872e04, 0.788731721804990e02),
)


def bc_v(teff: float) -> float:
    lt = math.log10(teff)
    c = BC_COEF[0] if lt < 3.70 else BC_COEF[1] if lt < 3.90 else BC_COEF[2]
    return sum(ci * lt**i for i, ci in enumerate(c))


MBOL_SUN = V_SUN + 31.572 + bc_v(TEFF_SUN)


def radius(v: float, plx_mas: float, teff: float) -> float:
    """Radius (Rsun) from V, parallax (mas) and Teff."""
    mbol = v + 5 * math.log10(plx_mas / 100) + bc_v(teff)
    return math.sqrt(10 ** (-0.4 * (mbol - MBOL_SUN))) * (TEFF_SUN / teff) ** 2


def simbad_at(ra: float, dec: float) -> dict | None:
    """The brightest SIMBAD object within 10 arcsec, with its V and parallax."""
    q = ("SELECT TOP 1 b.main_id, f.flux AS vmag, b.plx_value, b.plx_err, b.plx_bibcode FROM basic AS b "
         "LEFT JOIN flux AS f ON f.oidref = b.oid AND f.filter = 'V' "
         f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, 0.0028)) = 1 ORDER BY vmag")
    rows = tap(SIMBAD, q)
    return rows[0] if rows else None


def main():
    rows = []
    for key, star in sorted(T.STARS.items()):
        if not (T.OUT / f"{key}.json").exists():
            continue
        s = simbad_at(star.ra, star.dec)
        if s and not s["vmag"] and key in V_EXTRA:
            s["vmag"] = V_EXTRA[key][0]
        if not s or not s["plx_value"] or not s["vmag"]:
            print(f"{key}: no V or parallax in SIMBAD", flush=True)
            continue
        v, plx, e_plx = float(s["vmag"]), float(s["plx_value"]), float(s["plx_err"] or "nan")
        r = radius(v, plx, star.teff)
        rows.append(dict(key=key, simbad=s["main_id"], teff=star.teff, V=v, plx_mas=plx, e_plx_mas=e_plx,
                         plx_ref=s["plx_bibcode"], bc_v=round(bc_v(star.teff), 3), radius_sb=round(r, 3),
                         e_radius_sb=round(r * e_plx / plx, 3) if e_plx == e_plx else None,
                         radius_adopted=star.radius if T.RADII_SOURCE not in star.source else None))
        print(f"{key:16s} V {v:5.2f} plx {plx:7.2f}+-{e_plx:.2f} BC {bc_v(star.teff):+.2f} R_SB {r:5.2f} "
              f"adopted {rows[-1]['radius_adopted']}", flush=True)
    ids = [r["simbad"] for r in rows]
    rows = [r for r in rows if ids.count(r["simbad"]) == 1]
    for lo, hi in TEFF_BANDS:
        band = [r for r in rows if lo <= r["teff"] < hi]
        q = sorted(r["radius_sb"] / r["radius_adopted"] for r in band if r["radius_adopted"])
        q = [x for x in q if 0.75 < x < 1.25]  # without the outliers (HD 29647, reddened: 0.18)
        med = q[len(q) // 2]
        print(f"Teff {lo}-{hi} K: R_SB / R_adopted median {med:.3f} over {len(q)} stars")
        for r in band:
            r["radius_cal"] = round(r["radius_sb"] / med, 3)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
