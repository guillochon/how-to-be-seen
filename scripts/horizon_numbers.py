"""Distances out to which a 10 h exposure reaches the S/N = 1000 floor, per host, species and instrument.

Writes research/horizon.json; quoted in Sect. 4 of the paper.

    uv run python scripts/horizon_numbers.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, detect, hosts, survey  # noqa: E402

T_EXP, CAP = 10 * 3600.0, 1000.0


def main():
    curves = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))
    out = {}
    for h in hosts.host_grid():
        MV = survey.absolute_v(h.teff)
        for sp, c in curves[h.key].items():
            if sp == "Tc II" and h.teff < 7000:
                continue
            for inst in detect.INSTRUMENTS:
                sel = [(ln, fr) for ln, fr in zip(c.lines, c.flux_ratio, strict=True) if inst.wmin <= ln.wl <= inst.wmax]
                if not sel:
                    continue
                # distance at which the median covered line reaches S/N = CAP in T_EXP
                ln, fr = sorted(sel, key=lambda t: t[0].wl)[len(sel) // 2]
                f = lambda logd, inst=inst, MV=MV, ln=ln, fr=fr: np.log10(  # noqa: E731
                    detect.snr_per_pixel(inst, MV + 5 * (logd - 1), T_EXP, ln.wl, fr) / CAP)
                d = 10 ** brentq(f, -3, 6)
                out[f"{h.key}|{sp}|{inst.name}"] = dict(d_floor_pc=d, V_floor=MV + 5 * np.log10(d / 10), MV=MV)
                print(f"{h.key:3s} {sp:6s} {inst.name:36s} floor to {d:8.1f} pc (V = {MV + 5*np.log10(d/10):5.2f})")
    (ROOT / "research" / "horizon.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
