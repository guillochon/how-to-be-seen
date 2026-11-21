"""List the archive collections and observing programmes behind the spectra of every star with a Tc limit, for the
paper's acknowledgements.

ESO asks that processed (Phase 3) data be acknowledged by the DOI of their collection
(https://archive.eso.org/wdb/wdb/doi/collections/query), and MAST that HST data list their programme numbers.
Reads the downloaded files in research/spectra/ and the ESO archive (TAP) for each file's collection.
Writes research/tc_limits/archive_programmes.json.

    uv run python scripts/archive_programmes.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

TAP = "https://archive.eso.org/tap_obs/sync"
DOIS = "https://archive.eso.org/wdb/wdb/doi/collections/query?wdbo=csv&max_rows_returned=1000"


def main():
    keys = [p.stem for p in T.OUT.glob("*.json") if p.stem in T.STARS and p.stem not in T.EXCLUDED
            and (T.SPECTRA / p.stem).exists()]
    ids = [re.sub(r"T(\d\d)_(\d\d)_", r"T\1:\2:", f.stem) for k in keys for inst in ("uves", "harps", "feros", "espresso")
           for f in (T.SPECTRA / k / inst).glob("ADP*.fits")]
    coll, prog = Counter(), set()
    for i in range(0, len(ids), 150):
        chunk = ",".join(f"'{x}'" for x in ids[i:i + 150])
        q = f"SELECT dp_id, obs_collection, proposal_id FROM ivoa.ObsCore WHERE dp_id IN ({chunk})"
        u = TAP + "?" + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
        for r in csv.DictReader(io.StringIO(urllib.request.urlopen(u, timeout=300).read().decode())):
            coll[r["obs_collection"]] += 1
            prog.update(p.strip() for p in r["proposal_id"].split(",") if p.strip())
    listing = urllib.request.urlopen(DOIS, timeout=300).read().decode().lstrip()  # it opens with a blank line
    doi = {r["Collection Name"]: r["DOI"] for r in csv.DictReader(io.StringIO(listing))}
    hst = set()
    for k in keys:
        for f in [*(T.SPECTRA / k / "stis").glob("*_x1d.fits"), *(T.SPECTRA / k / "ghrs").glob("*_c1f.fits")]:
            h = fits.getheader(f)
            hst.add(str(h.get("PROPOSID") or ""))
    hst.discard("")
    out = dict(eso_files=len(ids), eso_collections={c: dict(files=n, doi=doi.get(c)) for c, n in coll.most_common()},
               eso_programmes=sorted(prog), hst_programmes=sorted(hst, key=int))
    (T.OUT / "archive_programmes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "eso_programmes"}, indent=1))


if __name__ == "__main__":
    main()
