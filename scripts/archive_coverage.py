"""High-resolution spectra of candidate Tc-search stars in the ESO and MAST archives.

Each name is resolved in SIMBAD (coordinates, V, spectral type). ESO Phase 3 spectra are counted per
instrument (UVES, HARPS, ESPRESSO, FEROS) with the summed S/N, sqrt(sum S/N^2), and MAST holdings are
counted for the near-ultraviolet settings that cover the Tc II lines at 2500-3250 A: HST STIS E230H and
E230M, GHRS echelle B and its G200M and G270M gratings (instrument "HRS" in MAST), and IUE LWP and LWR
high dispersion.

    uv run python scripts/archive_coverage.py "HD 101065" "Sirius" ...   # writes research/tc_limits/archive_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT  # noqa: E402

SIMBAD = "https://simbad.harvard.edu/simbad/sim-tap/sync"  # mirror of CDS
TAP = "https://archive.eso.org/tap_obs/sync"
MAST = "https://mast.stsci.edu/api/v0/invoke"
ESO_INST = ("UVES", "HARPS", "ESPRESSO", "FEROS")
RADIUS = 0.005  # deg


def _read(url: str, data: bytes | None = None, tries: int = 3) -> bytes:
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, data, timeout=300).read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError


def tap(url: str, q: str) -> list[dict]:
    u = url + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    return list(csv.DictReader(io.StringIO(_read(u).decode())))


def simbad(name: str) -> dict:
    q = ("SELECT b.main_id, b.ra, b.dec, b.sp_type, f.flux AS vmag FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
         "LEFT JOIN flux AS f ON f.oidref = b.oid AND f.filter = 'V' WHERE i.id = '" + name.replace("'", "''") + "'")
    rows = tap(SIMBAD, q)
    if not rows:
        raise KeyError(name)
    return rows[0]


def eso(ra: float, dec: float) -> dict:
    q = ("SELECT instrument_name, snr FROM ivoa.ObsCore WHERE dataproduct_type='spectrum' AND "
         f"CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {RADIUS}))=1")
    out = {}
    for r in tap(TAP, q):
        if r["instrument_name"] in ESO_INST:
            n, s2 = out.get(r["instrument_name"], (0, 0.0))
            out[r["instrument_name"]] = (n + 1, s2 + float(r["snr"] or 0) ** 2)
    return {k: (n, s2**0.5) for k, (n, s2) in out.items()}


def mast(ra: float, dec: float) -> dict:
    req = {"service": "Mast.Caom.Filtered.Position", "format": "json", "pagesize": 5000,
           "params": {"columns": "obs_collection,instrument_name,filters,dataproduct_type",
                      "filters": [{"paramName": "obs_collection", "values": ["HST", "IUE"]},
                                  {"paramName": "dataproduct_type", "values": ["spectrum"]}],
                      "position": f"{ra}, {dec}, {RADIUS}"}}
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    for _ in range(30):  # the MAST API answers EXECUTING until the query is done
        res = json.loads(_read(MAST, data))
        if res.get("status") == "COMPLETE" and "data" in res:
            break
        time.sleep(5)
    else:
        raise OSError(f"MAST query not complete: {res.get('status')} {res.get('msg', '')}")
    rows = res["data"]
    out = {"STIS E230": 0, "GHRS NUV": 0, "IUE LW high": 0}
    for r in rows:
        inst, filt = r["instrument_name"] or "", r["filters"] or ""
        if r["obs_collection"] == "HST" and inst.startswith("STIS") and filt.startswith("E230"):
            out["STIS E230"] += 1
        elif r["obs_collection"] == "HST" and inst.startswith("HRS") and filt in ("ECH-B", "G200M", "G270M"):
            out["GHRS NUV"] += 1
        elif r["obs_collection"] == "IUE" and inst in ("LWP", "LWR") and "HIGH" in filt:
            out["IUE LW high"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+")
    names = ap.parse_args().names
    rows = []
    for name in names:
        try:
            s = simbad(name)
        except KeyError:
            print(f"{name}: not found in SIMBAD", flush=True)
            continue
        ra, dec = float(s["ra"]), float(s["dec"])
        try:
            e, m = eso(ra, dec), mast(ra, dec)
        except OSError as err:
            print(f"{name}: archive query failed ({err})", flush=True)
            continue
        row = dict(name=name, simbad=s["main_id"], ra=ra, dec=dec, V=s["vmag"], sptype=s["sp_type"])
        for k in ESO_INST:
            n, snr = e.get(k, (0, 0.0))
            row[f"{k}_n"], row[f"{k}_snr"] = n, round(snr)
        row.update(m)
        rows.append(row)
        print(f"{name:14s} {s['main_id']:22s} V={s['vmag'] or '?':>5.5} {s['sp_type']:14.14s} " +
              " ".join(f"{k} {row[f'{k}_n']}({row[f'{k}_snr']})" for k in ESO_INST) +
              " | " + " ".join(f"{k} {v}" for k, v in m.items()), flush=True)
    path = ROOT / "research" / "tc_limits" / "archive_coverage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
