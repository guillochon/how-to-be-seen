"""Download the public archival spectra of a star in scripts/tc_limits.py STARS.

    ESO    reduced (Phase 3) UVES, HARPS, FEROS and ESPRESSO spectra with S/N >= ESO_MIN_SNR (UVES: those
           whose range starts below 5000 A), at most ESO_MAX of the best per instrument
    STIS   every HST STIS E230H and E230M echelle exposure in MAST (the x1d product)
    GHRS   every HST GHRS echelle-B and G270M exposure in MAST that covers GHRS_RANGE (calibrated c0f
           wavelengths, c1f fluxes and c2f errors)
    IUE    every high-dispersion long-wavelength (LWR, LWP) exposure in MAST
    PolarBase  ESPaDOnS and Narval spectra (intensity, .s files) by HD number, at most POLAR_MAX of the
           best per instrument
    CADC   ESPaDOnS intensity spectra (Upena i.fits) of the CFHT archive that PolarBase lacks, the longest
           exposures first, up to POLAR_MAX ESPaDOnS spectra in all
    OHP    ELODIE and SOPHIE (HR and HE modes) 1D spectra from the Haute-Provence archive, by HD number, with
           S/N >= ESO_MIN_SNR, at most ESO_MAX of the best per instrument

Files land in research/spectra/<star>/{uves,harps,feros,espresso,stis,ghrs,iue,elodie,sophie_hr,sophie_he,
espadons,narval}; existing files are skipped, and a CSV of the
ESO selection is written beside them.

    uv run python scripts/fetch_spectra.py hd101065 [more stars]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tc_limits import SPECTRA, STARS  # noqa: E402

TAP = "https://archive.eso.org/tap_obs/sync"
ESO_FILE = "https://dataportal.eso.org/dataportal_new/file/"
MAST = "https://mast.stsci.edu/api/v0/invoke"
HST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{obs}_{ext}.fits"
GHRS_RANGE = (2280.0, 3300.0)  # A, the near-ultraviolet Tc II lines
POLARBASE = "https://polarbase.irap.omp.eu/"
POLAR_MAX = 20  # each spectrum is a 20 MB text file
CADC_TAP = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
CADC_FILE = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/{product}.fits"
OHP = "http://atlas.obs-hp.fr/"
ELODIE_LIST = OHP + "elodie/fE.cgi?ob=objname,dataset,imanum&c=o&o={name}"
ELODIE_FILE = OHP + "elodie/fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:{dataset}/{imanum}"
SOPHIE_LIST = OHP + "sophie/sophie.cgi?n=sophie&a=htab&ob=seq&c=o&o={name}"
SOPHIE_FILE = OHP + "sophie/sophie.cgi?c=i&a=mime:application/fits&o=sophie:[s1d,{seq}]"
IUE = "https://archive.stsci.edu/missions/iue/data/{cam}/{grp}/{obs}.mxhi.gz"
RADIUS = 0.01  # deg
ESO_MIN_SNR, ESO_MAX = 30.0, 60


def eso_rows(ra: float, dec: float) -> list[dict]:
    q = ("SELECT instrument_name, dp_id, t_min, em_min, em_max, em_res_power, snr FROM ivoa.ObsCore "
         "WHERE dataproduct_type='spectrum' AND CONTAINS(POINT('ICRS', s_ra, s_dec), "
         f"CIRCLE('ICRS', {ra}, {dec}, {RADIUS}))=1")
    url = TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    return list(csv.DictReader(io.StringIO(urllib.request.urlopen(url, timeout=120).read().decode())))


def mast_obs(ra: float, dec: float) -> list[dict]:
    """HST and IUE spectra within RADIUS; the MAST API answers EXECUTING until the query is done."""
    req = {"service": "Mast.Caom.Filtered.Position", "format": "json", "pagesize": 5000,
           "params": {"columns": "obs_id,obs_collection,instrument_name,filters",
                      "filters": [{"paramName": "obs_collection", "values": ["HST", "IUE"]},
                                  {"paramName": "dataproduct_type", "values": ["spectrum"]}],
                      "position": f"{ra}, {dec}, {RADIUS}"}}
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    for _ in range(40):
        res = json.loads(urllib.request.urlopen(MAST, data, timeout=300).read())
        if res.get("status") == "COMPLETE" and "data" in res:
            return res["data"]
        time.sleep(5)
    raise OSError(f"MAST query not complete: {res.get('status')}")


def iue_obs(rows: list[dict]) -> list[str]:
    return sorted({r["obs_id"] for r in rows if r["obs_collection"] == "IUE" and r["instrument_name"] in ("LWR", "LWP")
                   and "HIGH" in (r["filters"] or "")})


def stis_obs(rows: list[dict]) -> list[str]:
    """Single STIS echelle exposures (the 9-character dataset names, not the merged HASP/HSLA products)."""
    return sorted({r["obs_id"] for r in rows if r["obs_collection"] == "HST" and (r["instrument_name"] or "").startswith("STIS")
                   and (r["filters"] or "") in ("E230H", "E230M") and len(r["obs_id"]) == 9})


def ghrs_obs(rows: list[dict]) -> list[str]:
    return sorted({r["obs_id"] for r in rows if r["obs_collection"] == "HST" and (r["instrument_name"] or "").startswith("HRS")
                   and (r["filters"] or "") in ("ECH-B", "G270M")})


def ohp_table(url: str) -> list[dict]:
    """Rows of an ELODIE or SOPHIE archive listing, as {column: text}."""
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("latin-1")
    except urllib.error.HTTPError:
        raise
    except (TimeoutError, urllib.error.URLError):
        return []  # the archive never answers for an object it holds no spectra of (HD 101065, HD 999999)
    heads = [re.sub("<[^>]*>", "", h).strip() for h in re.findall(r"<th.*?</th>", html, re.S)]
    rows = []
    for tr in re.findall(r"<tr>.*?</tr>", html, re.S | re.I):
        cells = [re.sub("<[^>]*>", "", c).strip() for c in re.findall(r"<td.*?</td>", tr, re.S | re.I)]
        if len(cells) >= 10:
            rows.append(dict(zip(heads, cells, strict=False)))
    return rows


def ohp_obs(name: str) -> dict[str, list[tuple[str, str]]]:
    """{instrument: [(url, file name)]} of the best ELODIE and SOPHIE spectra of an HD star."""
    if not _hd(name):
        return {}
    hd = urllib.parse.quote(_hd(name))
    out = {}
    el = [r for r in ohp_table(ELODIE_LIST.format(name=hd)) if r.get("imatyp", "").startswith("OBJ")
          and float(r.get("sn") or 0) >= ESO_MIN_SNR]
    el = sorted(el, key=lambda r: -float(r["sn"]))[:ESO_MAX]
    out["elodie"] = [(ELODIE_FILE.format(**r), f"elodie_{r['dataset']}_{r['imanum']}.fits") for r in el]
    so = [r for r in ohp_table(SOPHIE_LIST.format(name=hd)) if float(r.get("sn26") or 0) >= ESO_MIN_SNR]
    for mode in ("HR", "HE"):
        sel = sorted((r for r in so if r.get("mode") == mode), key=lambda r: -float(r["sn26"]))[:ESO_MAX]
        out[f"sophie_{mode.lower()}"] = [(SOPHIE_FILE.format(**r), f"sophie_{r['seq']}.fits") for r in sel]
    return out


def _hd(name: str) -> str | None:
    m = re.search(r"HD\s*(\d+)", name)
    return f"HD {int(m.group(1))}" if m else None


def _retry(fn, *args, tries: int = 4):
    """fn(*args), retried with growing pauses: the OHP and PolarBase servers time out or answer 502 at times."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(30 * (attempt + 1))
    raise AssertionError


def _post(url: str, payload: dict) -> bytes:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return _retry(lambda: urllib.request.urlopen(req, timeout=120).read())


def polarbase_obs(name: str) -> dict[str, list[tuple[str, str]]]:
    """{instrument: [(url, file name)]} of the best ESPaDOnS and Narval intensity spectra of an HD star."""
    hd = _hd(name)
    if not hd:
        return {}
    try:
        req = urllib.request.Request(POLARBASE + "api/spectra", json.dumps({"object_name": hd}).encode(),
                                     {"Content-Type": "application/json"})
        recs = json.loads(urllib.request.urlopen(req, timeout=120).read()).get("records", [])
    except urllib.error.HTTPError as e:
        if e.code in (400, 502):  # a name PolarBase does not hold: 400, or 502 once its resolver gives up (HD 999999)
            return {}
        raise
    out = {}
    for inst in ("espadons", "narval"):
        sel = sorted((r for r in recs if inst in r["instrument"] and float(r["snrmax"] or 0) >= ESO_MIN_SNR),
                     key=lambda r: -float(r["snrmax"]))[:POLAR_MAX]
        if not sel:
            continue
        links = _post(POLARBASE + "download/wget_links", {"observation_ids": [r["id_observation"] for r in sel],
                                                          "file_extensions": ["any"], "original_file": False})
        urls = [ln.split()[-1] for ln in links.decode().splitlines()  # normalized only, not the "_u" files
                if ln.strip().endswith(".s") and not ln.strip().endswith("_u.s")]
        out[inst] = [(u, u.rsplit("/", 1)[-1]) for u in urls]
    return out


def cadc_obs(ra: float, dec: float) -> list[str]:
    """Product IDs of the ESPaDOnS intensity spectra within RADIUS in the CFHT archive at CADC, longest first."""
    q = ("SELECT p.productID, p.time_exposure FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID = p.obsID "
         "WHERE o.instrument_name = 'ESPaDOnS' AND p.calibrationLevel = 2 AND p.productID LIKE '%i' "
         f"AND INTERSECTS(CIRCLE('ICRS', {ra}, {dec}, {RADIUS}), p.position_bounds) = 1")
    url = CADC_TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(url, timeout=300).read().decode())))
    return [r["productID"] for r in sorted(rows, key=lambda r: -float(r["time_exposure"] or 0))]


def fetch(url: str, path: Path) -> bool:
    """Download url to path unless it is there; False if the archive has no such file."""
    if path.exists() and path.stat().st_size:
        return True
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(url, timeout=300).read()
            if not data:  # SOPHIE answers some requests with an empty body
                print(f"missing {path.name}: empty", flush=True)
                path.unlink(missing_ok=True)
                return False
            path.write_bytes(data)
            print(f"{path.name}  {len(data) / 1e6:.1f} MB", flush=True)
            return True
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                print(f"missing {path.name}: {e}", flush=True)
                return False
            print(f"retry {path.name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
        except OSError as e:
            print(f"retry {path.name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def fetch_star(key: str) -> None:
    star = STARS[key]
    out = SPECTRA / key
    rows = eso_rows(star.ra, star.dec)
    sel = {}
    for inst in ("UVES", "HARPS", "FEROS", "ESPRESSO"):
        ok = [r for r in rows if r["instrument_name"] == inst and float(r["snr"] or 0) >= ESO_MIN_SNR
              and (inst != "UVES" or float(r["em_min"]) < 5.0e-7)]
        sel[inst.lower()] = sorted(ok, key=lambda r: -float(r["snr"]))[:ESO_MAX]
    for sub, chosen in sel.items():
        (out / sub).mkdir(parents=True, exist_ok=True)
        if not chosen:
            continue
        with (out / f"{sub}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chosen[0]))
            w.writeheader()
            w.writerows(chosen)
        for r in chosen:
            fetch(ESO_FILE + r["dp_id"], out / sub / (r["dp_id"].replace(":", "_") + ".fits"))
    found = {}
    for lookup in (ohp_obs, polarbase_obs):
        try:
            found.update(lookup(star.name))
        except (OSError, ValueError) as e:  # one archive's trouble should not stop the others
            print(f"{key}: {lookup.__name__} failed ({e})", flush=True)
    for sub, files in found.items():
        (out / sub).mkdir(parents=True, exist_ok=True)
        sel[sub] = [f for url, f in files if fetch(url, out / sub / f)]
    try:
        have = {f.name[:7] for f in (out / "espadons").glob("*")} if (out / "espadons").exists() else set()
        extra = [p for p in cadc_obs(star.ra, star.dec) if p[:7] not in have][:max(POLAR_MAX - len(have), 0)]
    except (OSError, ValueError) as e:
        print(f"{key}: cadc_obs failed ({e})", flush=True)
        extra = []
    if extra:
        (out / "espadons").mkdir(parents=True, exist_ok=True)
        got = [p for p in extra if fetch(CADC_FILE.format(product=p), out / "espadons" / f"{p}.fits")]
        sel["espadons"] = sel.get("espadons", []) + got
    rows = mast_obs(star.ra, star.dec)
    (out / "stis").mkdir(parents=True, exist_ok=True)
    stis = stis_obs(rows)
    stis = [obs for obs in stis if fetch(HST_FILE.format(obs=obs, ext="x1d"), out / "stis" / f"{obs}_x1d.fits")]
    (out / "ghrs").mkdir(parents=True, exist_ok=True)
    ghrs = []
    for obs in ghrs_obs(rows):
        c0 = out / "ghrs" / f"{obs}_c0f.fits"
        if not fetch(HST_FILE.format(obs=obs, ext="c0f"), c0):
            continue
        with fits.open(c0) as h:
            w = h[0].data
        if np.nanmax(w) > GHRS_RANGE[0] and np.nanmin(w) < GHRS_RANGE[1]:
            ghrs.append(obs)
            if not all([fetch(HST_FILE.format(obs=obs, ext=ext), out / "ghrs" / f"{obs}_{ext}.fits") for ext in ("c1f", "c2f")]):
                ghrs.pop()
    (out / "iue").mkdir(parents=True, exist_ok=True)
    iue = iue_obs(rows)
    for obs in iue:
        cam, num = obs[:3].lower(), obs[3:]
        fetch(IUE.format(cam=cam, grp=f"{int(num) // 1000 * 1000:05d}", obs=obs.lower()),
              out / "iue" / f"{obs.lower()}.mxhi.gz")
    print(f"{key}: " + ", ".join(f"{len(v)} {k.upper()}" for k, v in sel.items()) + ", "
          f"{len(stis)} STIS, {len(ghrs)} GHRS, {len(iue)} IUE in {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stars", nargs="+", choices=sorted(STARS))
    for key in ap.parse_args().stars:
        try:
            fetch_star(key)
        except (OSError, RuntimeError) as e:  # one star's archive trouble should not stop a batch
            print(f"{key}: failed ({e})", flush=True)


if __name__ == "__main__":
    main()
