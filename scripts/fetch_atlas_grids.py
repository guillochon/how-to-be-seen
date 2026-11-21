"""Download the metal-poor ATLAS9 grids of Castelli & Kurucz into research/kurucz/atm/.

scripts/tc_limits.py models a star in the grid nearest its [Fe/H]: the scaled-solar grids (am05..am25k2odfnew) for
[Fe/H] > -1 and the alpha-enhanced ones ([alpha/Fe] = +0.4; am10a..am25ak2odfnew) for the halo stars at or below.
The solar grid ap00k2odfnew.dat is already there. Existing files are skipped.

    uv run python scripts/fetch_atlas_grids.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ  # noqa: E402

BASE = "https://wwwuser.oats.inaf.it/fiorella.castelli/grids/grid{name}/a{name}.dat"
GRIDS = [f"m{m:02d}{a}k2odfnew" for m in (5, 10, 15, 20, 25) for a in ("", "a")]


def main():
    out = KURUCZ / "atm"
    out.mkdir(parents=True, exist_ok=True)
    for name in GRIDS:
        path = out / f"a{name}.dat"
        if path.exists() and path.stat().st_size:
            continue
        data = urllib.request.urlopen(BASE.format(name=name), timeout=300).read()
        path.write_bytes(data)
        print(f"{path.name}  {len(data) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
