"""Stars from the catalogue of chemically peculiar stars of Ghazaryan et al. (2018) that the Tc search can use.

The catalogue (research/ghazaryan2018/cpstars.tsv, VizieR J/MNRAS/480/2953) compiles published Teff, log g,
v sin i and abundances, rescaled to the solar abundances of Asplund et al. (2009). A star is kept if it has
all four (an HgMn star also without Fe) with v sin i <= VSINI_MAX and Teff in the ATLAS9 range used here. Each kept star is resolved in
SIMBAD, its ESO spectra are counted (UVES, HARPS, FEROS, ESPRESSO) and its Gaia DR3 radius (FLAME) is looked
up. The stars of the hand-compiled lists in LISTS are added the same way: the chemically normal late-B stars
(research/normal_lateb.md) and stars with near-ultraviolet echelle spectra in MAST (research/uv_candidates.md)
or UVES spectra below 3300 A (research/uves_candidates.md), and slowly rotating metal-poor dwarfs
(research/mp_candidates.md).
The result, research/tc_limits/cp_sample.csv, is read by scripts/tc_limits.py.

    uv run python scripts/cp_sample.py
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_coverage import _read, eso, simbad  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT  # noqa: E402

CATALOGUE = ROOT / "research" / "ghazaryan2018" / "cpstars.tsv"
REFS = ROOT / "research" / "ghazaryan2018" / "refs.tsv"
LISTS = ("normal_lateb", "uv_candidates", "uves_candidates", "mp_candidates")  # research/tc_limits/<name>.csv, documented in research/<name>.md
OUT = ROOT / "research" / "tc_limits" / "cp_sample.csv"
GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
FE_SUN = 7.50  # Asplund et al. (2009), the catalogue's solar scale
VSINI_MAX = 30.0  # km/s: broader lines blend away the weak Tc lines
TEFF_RANGE = (6000, 18000)


def _rows(path: Path) -> list[list[str]]:
    rows = [r for r in csv.reader(path.open(encoding="utf-8"), delimiter="\t") if r and not r[0].startswith("#")]
    return [[c.strip() for c in r] for r in rows[3:]]  # skip the header, units and dashes


def catalogue() -> list[dict]:
    refs = {r[1]: r[2] for r in _rows(REFS) if len(r) > 2}
    out = []
    for star, cp, teff, logg, vsini, fe, e_fe, r_fe in (r[:8] for r in _rows(CATALOGUE) if len(r) >= 8):
        if not (teff and logg and vsini and (fe or cp == "HgMn")):  # HgMn stars are the cheapest hosts: kept without Fe
            continue
        if float(vsini) > VSINI_MAX or not TEFF_RANGE[0] <= float(teff) <= TEFF_RANGE[1]:
            continue
        out.append(dict(star=star, cp=cp, teff=float(teff), logg=float(logg), vsini=float(vsini),
                        fe=FE_SUN + float(fe) if fe else None, e_fe=float(e_fe) if e_fe else None, fe_ref=refs.get(r_fe, "")))
    return out


def listed() -> list[dict]:
    """The stars of the hand-compiled LISTS; Fe there is already log eps. The normal late-B list has no cp column."""
    out = []
    for name in LISTS:
        path = ROOT / "research" / "tc_limits" / f"{name}.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            out.append(dict(star=r["star"], cp=r.get("cp") or "normal", teff=float(r["teff"]), logg=float(r["logg"]),
                            vsini=float(r["vsini"]), fe=float(r["fe"]) if r["fe"] else None,
                            e_fe=float(r["e_fe"]) if r["e_fe"] else None, fe_ref=r["ref_fe"], params_ref=r["ref_params"],
                            list=name))
    return out


_gaia_down = False


def gaia_radius(ra: float, dec: float) -> float | None:
    q = ("SELECT a.radius_flame FROM gaiadr3.gaia_source p JOIN gaiadr3.astrophysical_parameters a USING (source_id) "
         f"WHERE CONTAINS(POINT('ICRS', p.ra, p.dec), CIRCLE('ICRS', {ra}, {dec}, 0.003)) = 1 ORDER BY p.phot_g_mean_mag")
    global _gaia_down
    if _gaia_down:
        return None
    u = GAIA + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    try:
        rows = list(csv.DictReader(io.StringIO(_read(u).decode())))
    except OSError as e:  # the Gaia archive is sometimes down (HTTP 503); scripts/radii.py then gives the radius
        print(f"Gaia radius query failed ({e}); no FLAME radius for this and later stars of this run", flush=True)
        _gaia_down = True  # each failure costs about 2 minutes of retries
        return None
    return float(rows[0]["radius_flame"]) if rows and rows[0]["radius_flame"] else None


def main():
    done = {}
    if OUT.exists():
        done = {r["star"]: r for r in csv.DictReader(OUT.open())}
    rows = []
    for c in catalogue() + listed():
        if c["star"] in done:
            rows.append(done[c["star"]])
            continue
        try:
            s = simbad(c["star"])
        except KeyError:
            print(f"{c['star']}: not in SIMBAD", flush=True)
            continue
        ra, dec = float(s["ra"]), float(s["dec"])
        e = eso(ra, dec)
        row = dict(c, simbad=s["main_id"], ra=ra, dec=dec, V=s["vmag"], radius=gaia_radius(ra, dec),
                   **{f"{k}_n": e.get(k, (0, 0))[0] for k in ("UVES", "HARPS", "FEROS", "ESPRESSO")},
                   **{f"{k}_snr": round(e.get(k, (0, 0))[1]) for k in ("UVES", "HARPS", "FEROS", "ESPRESSO")})
        rows.append(row)
        print(f"{c['star']:22s} {c['cp']:9s} {c['teff']:6.0f} Fe {c['fe']} R {row['radius']} "
              f"UVES {row['UVES_n']} HARPS {row['HARPS_n']} FEROS {row['FEROS_n']} ESPRESSO {row['ESPRESSO_n']}", flush=True)
        write(rows, done)  # as it goes, so an interrupted run resumes
    write(rows, done)


def write(rows: list[dict], done: dict[str, dict]) -> None:
    """The sample so far, with the rows of an earlier run not yet reached."""
    seen = {r["star"] for r in rows}
    with OUT.open("w", newline="") as f:
        rows = rows + [r for k, r in done.items() if k not in seen]
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
