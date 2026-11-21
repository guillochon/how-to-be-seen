"""Summarize research/tc_limits/detect.csv (scripts/tc_detect.py) by host class.

For each star: the lowest detectable abundance over its spectra with clean lines only, the lowest with photon
noise alone and every line, and the archival limit. The gap between the first two is how much a real detection
costs over the ideal of the paper's thresholds; a gap of g dex raises the inventory by 10^g and the Sagan type by
g / 10. Stars with no clean Tc line in any spectrum are counted apart.

    uv run python scripts/tc_detect_summary.py     # writes research/tc_limits/detect_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402


def f(x):
    return float(x) if x not in ("", "None", None) else None


def main():
    rows = list(csv.DictReader((T.OUT / "detect.csv").open(encoding="utf-8")))
    stars = {}
    for r in rows:
        s = stars.setdefault(r["key"], dict(name=r["name"], cp=r["cp"], teff=float(r["teff"]), photon=[], clean=[], limit=[]))
        if f(r["le_photon"]) is not None:
            s["photon"].append(f(r["le_photon"]))
        if f(r["le_clean"]) is not None:
            s["clean"].append(f(r["le_clean"]))
        if f(r["limit"]) is not None:
            s["limit"].append(f(r["limit"]))
    out = {}
    for key, s in stars.items():
        star = T.STARS[key]
        cls = T.host_class(key, star)
        o = out.setdefault(cls, dict(n=0, no_clean=[], gaps=[], clean_minus_limit=[]))
        o["n"] += 1
        if not s["clean"] or not s["photon"]:
            o["no_clean"].append(s["name"])
            continue
        o["gaps"].append(min(s["clean"]) - min(s["photon"]))
        if s["limit"]:
            o["clean_minus_limit"].append(min(s["clean"]) - min(s["limit"]))
    for cls, o in out.items():
        g = np.array(o["gaps"])
        cl = np.array(o["clean_minus_limit"])
        o.update(median_gap=float(np.median(g)) if len(g) else None,
                 gap_16_84=[float(x) for x in np.percentile(g, [16, 84])] if len(g) else None,
                 median_clean_minus_limit=float(np.median(cl)) if len(cl) else None,
                 n_no_clean=len(o["no_clean"]))
        print(f"{cls:22s} stars {o['n']:3d}  no clean line {o['n_no_clean']:3d}  median gap {o['median_gap']:.2f} dex "
              f"(16-84%: {o['gap_16_84'][0]:.2f} to {o['gap_16_84'][1]:.2f})  detection minus limit "
              f"{o['median_clean_minus_limit'] if o['median_clean_minus_limit'] is None else round(o['median_clean_minus_limit'], 2)}")
    (T.OUT / "detect_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
