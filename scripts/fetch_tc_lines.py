"""Stream Kurucz's computed Tc I and Tc III line lists and keep the lines that matter for radiative
accelerations in A and late-B atmospheres.

Writes research/kurucz/tc/tc{0,2}_lines.csv (wavelength in Angstrom, vacuum below 2000 A and air
above as in Kurucz's files, log gf, lower-level energy in cm^-1, upper-level energy, J_low,
predicted flag) and copies the partition-function tables partfn430{0,1,2}.dat.

Kept: log gf > -3, 900 A < wavelength < 20000 A, lower level below 80,000 cm^-1 (10 eV).
Levels with negative energies in the files are predicted, not measured.

    uv run python scripts/fetch_tc_lines.py
"""

from __future__ import annotations

import gzip
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "kurucz" / "tc"
BASE = "http://kurucz.harvard.edu/atoms"
SOURCES = {0: f"{BASE}/4300/gf4300.80col-gz", 2: f"{BASE}/4302/gf4302.80col-gz"}
LOGGF_MIN, WMIN, WMAX, ELOW_MAX = -3.0, 900.0, 20000.0, 80000.0


def stream(ion: int) -> int:
    n = 0
    with urllib.request.urlopen(SOURCES[ion]) as resp, gzip.GzipFile(fileobj=resp) as gz, \
            (OUT / f"tc{ion}_lines.csv").open("w") as out:
        out.write("wl_A,loggf,elow_cm,eupp_cm,jlow,predicted\n")
        for raw in gz:
            ln = raw.decode("ascii", "replace")
            try:
                wl = 10 * float(ln[0:11])
                loggf = float(ln[11:18])
                e1, j1 = float(ln[24:36]), float(ln[36:41])
                e2, j2 = float(ln[52:64]), float(ln[64:69])
            except ValueError:
                continue
            if loggf <= LOGGF_MIN or not WMIN < wl < WMAX:
                continue
            (el, jl, pl), (eu, _, pu) = sorted([(abs(e1), j1, e1 < 0), (abs(e2), j2, e2 < 0)])
            if el > ELOW_MAX:
                continue
            out.write(f"{wl:.4f},{loggf:.3f},{el:.3f},{eu:.3f},{jl:.1f},{int(pl or pu)}\n")
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ion in (int(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else (0, 2):
        print(f"Tc {'I' * (ion + 1)}: kept {stream(ion)} lines", flush=True)
    for ion in (0, 1, 2):
        urllib.request.urlretrieve(f"{BASE}/430{ion}/partfn430{ion}.dat", OUT / f"partfn430{ion}.dat")


if __name__ == "__main__":
    main()
