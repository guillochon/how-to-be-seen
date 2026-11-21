"""How much the crowded solar spectrum costs: thresholds from clean beacon lines only.

A line is clean if the Kurucz (2005) solar flux atlas stays above 0.9 of the continuum within
+/- 0.05 A of its center. Writes research/blending.json; quoted in Sect. 4.

    uv run python scripts/blending.py
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ, ROOT, detect, hosts, survey  # noqa: E402

CLEAN = 0.9
HALF = 0.05


def main():
    raw = np.loadtxt(KURUCZ / "sun" / "solarfluxintwl.asc", skiprows=14)
    wl, fl = 10.0 * raw[:, 0], raw[:, 1]
    curves = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))
    inst = {i.name: i for i in detect.INSTRUMENTS}["8.2 m, R = 140,000"]
    out = {}
    for sp in ("Tc I", "Pm II"):
        c = curves["G"][sp]
        fmin = []
        for ln in c.lines:
            sel = np.abs(wl - ln.wl) < HALF
            fmin.append(float(fl[sel].min()) if sel.any() else np.nan)
        keep = [i for i, f in enumerate(fmin) if f > CLEAN]
        cc = copy.copy(c)
        cc.lines = [c.lines[i] for i in keep]
        cc.ew, cc.fwhm_int, cc.flux_ratio = c.ew[keep], c.fwhm_int[keep], c.flux_ratio[keep]
        rec = dict(n_lines=len(c.lines), n_clean=len(keep), clean=[c.lines[i].label for i in keep],
                   atlas_min=dict(zip([ln.label for ln in c.lines], fmin, strict=True)))
        for snr in (300.0, 1000.0):
            le_all = survey.required_log_eps(c, inst, snr)
            le_clean = survey.required_log_eps(cc, inst, snr) if keep else np.inf
            rec[f"{snr:.0f}"] = dict(log_eps_all=le_all, log_eps_clean=le_clean)
        out[sp] = rec
        print(sp, rec["n_clean"], "of", rec["n_lines"], "clean:", rec["clean"])
        print("   ", {k: v for k, v in rec.items() if k in ("300", "1000")})
    (ROOT / "research" / "blending.json").write_text(json.dumps(out, indent=1, default=float))
    _ = hosts


if __name__ == "__main__":
    main()
