"""Fetch Kurucz's gfall line lists and partition functions for every element, and NIST ionization energies and
atomic weights, for full spectral synthesis of chemically peculiar stars (scripts/model_residuals.py).

Writes, without touching files already present:
- research/kurucz/gfall/gfZZII.all for Z = 2-92 and ions I-III (Kurucz, linelists/gfall)
- research/kurucz/partfn/partfnZZII.dat where Kurucz has one (atoms/ZZII)
- research/kurucz/nist_ie.csv: Z, ion, ground level, ionization energy (eV), from the NIST ASD
- research/kurucz/nist_weights.csv: Z, standard atomic weight, from NIST Atomic Weights and Isotopic Compositions

    uv run python scripts/fetch_gfall.py
"""

from __future__ import annotations

import csv
import io
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KZ = ROOT / "research" / "kurucz"
BASE = "http://kurucz.harvard.edu"
NIST_IE = ("https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra={}&units=1&format=2&order=0&at_num_out=on"
           "&sp_name_out=on&ion_charge_out=on&level_out=on&e_out=0")
NIST_W = "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some"


def get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "isotope-beacons/1.0 (research script)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError:
        return None


def main():
    index = get(f"{BASE}/linelists/gfall/").decode("ascii", "replace")
    listed = set(re.findall(r"gf(\d{4})\.all", index))
    n_gf = n_pf = 0
    for z in range(2, 93):
        for ion in range(3):
            s = f"{z:02d}{ion:02d}"
            out = KZ / "gfall" / f"gf{s}.all"
            if s in listed and not out.exists():
                data = get(f"{BASE}/linelists/gfall/gf{s}.all")
                if data:
                    out.write_bytes(data)
                    n_gf += 1
            pf = KZ / "partfn" / f"partfn{s}.dat"
            if not pf.exists() and z != 43:
                data = get(f"{BASE}/atoms/{s}/partfn{s}.dat")
                if data:
                    pf.write_bytes(data)
                    n_pf += 1
    print(f"fetched {n_gf} line lists and {n_pf} partition-function tables")

    rows = []
    txt = get(NIST_IE.format("H-U")).decode("utf-8", "replace")
    for r in csv.reader(io.StringIO(txt)):
        v = [c.strip('="') for c in r]
        if len(v) < 6 or not v[0].isdigit():
            continue
        ion = int(v[2].lstrip("+"))
        if ion <= 3 and v[5]:
            rows.append((int(v[0]), ion, v[3], float(v[5])))
    with (KZ / "nist_ie.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z", "ion", "ground", "ie_ev"])
        w.writerows(rows)

    txt = get(NIST_W).decode("utf-8", "replace")
    weights = {}
    for block in txt.split("\n\n"):
        z = re.search(r"Atomic Number = (\d+)", block)
        aw = re.search(r"Standard Atomic Weight = \[?([\d.]+)", block)
        if z and aw:
            weights.setdefault(int(z.group(1)), float(aw.group(1)))
        elif z and int(z.group(1)) not in weights:
            m = re.search(r"Relative Atomic Mass = ([\d.]+)", block)
            if m:
                weights[int(z.group(1))] = float(m.group(1))
    with (KZ / "nist_weights.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z", "weight"])
        w.writerows(sorted(weights.items()))
    print(f"NIST: {len(rows)} ionization energies, {len(weights)} atomic weights")


if __name__ == "__main__":
    main()
