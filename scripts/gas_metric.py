"""The gas beacons costed by the probability that a monitoring program of given length detects them.

Sections 5 and 7.1 cost each gas beacon by the event rate f that lets a program taking n spectra a year catch one
spectrum during an event each year on average: f = 1 / (n t), with t the time an event stays detectable. For long
events (n t >> 1) the caught spectra come in clumps, so a short program most likely sees none (referee comment
2.6); for short events the average is a stricter demand than a detection. Here f is set instead by the probability P
that a program of T years detects at least one event. With events arriving at random at rate f, each caught by at
least one spectrum with probability p = 1 - exp(-n t),
    P = 1 - exp(-f T p),   so   f = -ln(1 - P) / (T p).
Every cost in research/comet.json, comet_opt.json and broadcast.json scales with f, so the Sagan type moves by
log10(f_new / f_old) / 10. The best artificial comet was optimized for the old metric, so its new cost is an upper
bound. For the doped comets read in Tc II from space we also give the cost at N_SPACE spectra a year, a cadence
that a space ultraviolet program could sustain on one star, instead of HARPS's 530.

    uv run python scripts/gas_metric.py     # writes research/gas_metric.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT  # noqa: E402
from beacon.const import YR  # noqa: E402

N_HARPS = 9000 / 17.0
N_SPACE = 20.0
T_PROG = 10.0  # yr
PS = (0.5, 0.9)


def shift(t_s: float, n: float, P: float, n_old: float = N_HARPS) -> float:
    """Change of Sagan type from the old metric (n_old spectra a year, one caught a year) to probability P."""
    f_old = YR / (n_old * t_s)
    p = -np.expm1(-n * t_s / YR)
    f_new = -np.log1p(-P) / (T_PROG * p)
    return float(np.log10(f_new / f_old) / 10)


def main():
    rows = []
    com = json.loads((ROOT / "research" / "comet.json").read_text())["design"]["rows"]
    for k, r in com.items():
        if "sagan_K" not in r:
            continue
        rec = dict(beacon=f"doped comet: {k}", t_s=r["t_transit_s"], K_old=r["sagan_K"])
        for P in PS:
            rec[f"K_P{P}"] = r["sagan_K"] + shift(r["t_transit_s"], N_HARPS, P)
            if "Tc II" in k:
                rec[f"K_P{P}_space{N_SPACE:.0f}"] = r["sagan_K"] + shift(r["t_transit_s"], N_SPACE, P)
        rows.append(rec)
    opt = json.loads((ROOT / "research" / "comet_opt.json").read_text())
    for k, r in [("artificial comet: best", opt["best"])] + [(f"artificial comet: {k}", v) for k, v in opt["rows"].items()]:
        rec = dict(beacon=k, t_s=r["t_eff_s"], K_old=r["sagan_K"], K_on=r.get("sagan_K_on"))
        for P in PS:
            rec[f"K_P{P}"] = r["sagan_K"] + shift(r["t_eff_s"], N_HARPS, P)
        rows.append(rec)
    bro = json.loads((ROOT / "research" / "broadcast.json").read_text())
    for r in bro["rows"]:
        rec = dict(beacon=f"belt/shell, theta {r['theta_deg']} deg", t_s=r["t_vis_s"], K_old=r["sagan_K"],
                   K_on=r.get("sagan_K_on"))
        for P in PS:
            rec[f"K_P{P}"] = r["sagan_K"] + shift(r["t_vis_s"], N_HARPS, P)
        rows.append(rec)
    for rec in rows:
        extra = "  ".join(f"{k} {v:.2f}" for k, v in rec.items() if k.startswith("K_") and k != "K_old" and v is not None)
        print(f"{rec['beacon']:42s} t = {rec['t_s'] / 86400:7.2f} d  K_old {rec['K_old']:.2f}  {extra}")
    (ROOT / "research" / "gas_metric.json").write_text(json.dumps(dict(n_harps=N_HARPS, n_space=N_SPACE,
                                                                      t_prog_yr=T_PROG, rows=rows), indent=1))


if __name__ == "__main__":
    main()
