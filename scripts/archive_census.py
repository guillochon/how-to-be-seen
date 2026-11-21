"""Census of the archives: which bright, slowly rotating late-B and A0 stars have high-resolution spectra.

The Tc sample (scripts/cp_sample.py) starts from the literature. This script starts from the archives instead, to
find stars that the sample misses. SIMBAD gives every star of type B3 to A0 with V < V_MAX and the mean and minimum
of its v sin i measurements (table mesRot). Supergiants and bright giants (class I and II) are dropped, and so are
stars whose smallest v sin i exceeds VSINI_MAX. For each remaining star the script counts
    ESO     Phase 3 spectra of UVES, HARPS, FEROS and ESPRESSO; UVES_3237 counts the UVES spectra that start
            below 3230 A and so reach Tc II 3237 A (the 390 nm setting starts at 3282 A and does not)
    MAST    HST STIS E230H and E230M, GHRS echelle-B, G200M and G270M, and IUE LWP and LWR high dispersion
    CADC    reduced ESPaDOnS intensity spectra
    KOA     Keck HIRES exposures
and writes research/tc_limits/archive_census.csv. Each line of the work file research/tc_limits/archive_census.jsonl
is one star, so an interrupted run resumes where it stopped. ELODIE, SOPHIE and PolarBase are searched only by
name and are not counted here. research/archive_census.md describes the result.

    uv run python scripts/archive_census.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT  # noqa: E402

SIMBAD = "https://simbad.harvard.edu/simbad/sim-tap/sync"  # mirror of CDS
ESO = "https://archive.eso.org/tap_obs/sync"
MAST = "https://mast.stsci.edu/api/v0/invoke"
CADC = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
KOA = "https://koa.ipac.caltech.edu/TAP/sync"
OUT = ROOT / "research" / "tc_limits" / "archive_census.csv"
WORK = OUT.with_suffix(".jsonl")
V_MAX = 8.5
VSINI_MAX = 30.0  # km/s, as in scripts/cp_sample.py
TC_3237_START = 3.23e-7  # m: a UVES spectrum must start below this to cover Tc II 3237 A
RADIUS = 0.005  # deg
TYPES = ("B3", "B4", "B5", "B6", "B7", "B8", "B9", "A0")


def _read(url: str, data: bytes | None = None, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, data, timeout=300).read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def tap(url: str, q: str) -> list[dict]:
    u = url + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q, MAXREC=200000))
    return list(csv.DictReader(io.StringIO(_read(u).decode())))


def candidates() -> list[dict]:
    types = " OR ".join(f"b.sp_type LIKE '{t}%'" for t in TYPES)
    q = ("SELECT b.oid, b.main_id, b.ra, b.dec, b.sp_type, f.flux AS v, AVG(r.vsini) AS vsini, MIN(r.vsini) AS vsini_min, "
         "COUNT(r.vsini) AS n_vsini FROM basic AS b JOIN flux AS f ON f.oidref = b.oid AND f.filter = 'V' "
         f"LEFT JOIN mesRot AS r ON r.oidref = b.oid WHERE f.flux < {V_MAX} AND ({types}) "
         "GROUP BY b.oid, b.main_id, b.ra, b.dec, b.sp_type, f.flux")
    rows = tap(SIMBAD, q)
    print(f"{len(rows)} stars of type B3-A0 with V < {V_MAX}; "
          f"{sum(not r['vsini'] for r in rows)} have no v sin i in SIMBAD", flush=True)

    def bright(sp: str) -> bool:  # luminosity class I or II, but not III or IV
        m = re.match(r"[BA][0-9.]*/?[0-9.]*\s*(\S*)", sp)
        lc = m.group(1) if m else ""
        return bool(re.match(r"(Ia|Ib|Iab|I$|I-|II$|II-|I/)", lc))

    return [r for r in rows if r["vsini"] and float(r["vsini_min"]) <= VSINI_MAX and not bright(r["sp_type"])]


def eso(ra: float, dec: float) -> dict:
    q = ("SELECT instrument_name, em_min FROM ivoa.ObsCore WHERE dataproduct_type = 'spectrum' AND "
         f"CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {RADIUS})) = 1")
    out = dict(UVES=0, UVES_3237=0, HARPS=0, FEROS=0, ESPRESSO=0)
    for r in tap(ESO, q):
        k = r["instrument_name"]
        if k in out:
            out[k] += 1
        if k == "UVES" and r["em_min"] and float(r["em_min"]) < TC_3237_START:
            out["UVES_3237"] += 1
    return out


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
        time.sleep(3)
    else:
        raise OSError(f"MAST query not complete: {res.get('status')}")
    out = {"STIS_E230": 0, "GHRS_NUV": 0, "IUE_LW_high": 0}
    for r in res["data"]:
        inst, filt = r["instrument_name"] or "", r["filters"] or ""
        if r["obs_collection"] == "HST" and inst.startswith("STIS") and filt.startswith("E230"):
            out["STIS_E230"] += 1
        elif r["obs_collection"] == "HST" and inst.startswith("HRS") and filt in ("ECH-B", "G200M", "G270M"):
            out["GHRS_NUV"] += 1
        elif r["obs_collection"] == "IUE" and inst in ("LWP", "LWR") and "HIGH" in filt:
            out["IUE_LW_high"] += 1
    return out


def cadc(ra: float, dec: float) -> dict:
    q = ("SELECT COUNT(*) AS n FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID = p.obsID "
         "WHERE o.instrument_name = 'ESPaDOnS' AND p.calibrationLevel = 2 AND p.productID LIKE '%i' "
         f"AND INTERSECTS(CIRCLE('ICRS', {ra}, {dec}, {RADIUS}), p.position_bounds) = 1")
    return {"ESPaDOnS": int(tap(CADC, q)[0]["n"])}


def koa(ra: float, dec: float) -> dict:
    q = f"SELECT COUNT(*) AS n FROM koa_hires WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01)) = 1"
    return {"HIRES": int(tap(KOA, q)[0]["n"])}


def census(star: dict) -> dict:
    ra, dec = float(star["ra"]), float(star["dec"])
    out = dict(star)
    for name, query in (("eso", eso), ("mast", mast), ("cadc", cadc), ("koa", koa)):
        try:
            out.update(query(ra, dec))
        except Exception as e:  # noqa: BLE001  one archive's failure is logged and the others still count
            out[f"{name}_error"] = str(e)[:80]
    return out


def main():
    stars = candidates()
    done = {json.loads(line)["oid"] for line in WORK.open()} if WORK.exists() else set()
    todo = [s for s in stars if s["oid"] not in done]
    print(f"{len(stars)} slow rotators; {len(done)} done, {len(todo)} to go", flush=True)
    lock, n = threading.Lock(), [0]

    def job(star: dict):
        out = census(star)
        with lock:
            n[0] += 1
            with WORK.open("a") as f:
                f.write(json.dumps(out) + "\n")
            errors = {k: v for k, v in out.items() if k.endswith("_error")}
            print(f"{n[0]}/{len(todo)} {out['main_id']} " + (f"ERROR {errors}" if errors else "ok"), flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(job, todo))
    rows = [json.loads(line) for line in WORK.open()]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
