"""Each archival star's own beacon threshold, in its own model atmosphere, and the matched-filter upper limits.

Referee comment 2.4: the archival limits were compared with the threshold of a representative host (the Sun, the Am
star, the quiet late-B star). Here every star gets the threshold of its class's beacon computed in the ATLAS9 model
nearest it (its [Fe/H] grid for metal-poor and HB stars; scripts/tc_limits.py synths) with its own v sin i, at
S/N = 300 per pixel: the three Tc I resonance lines with the 8.2 m, R = 140,000 instrument for the stars that mix
their convection zone, and the Tc II lines with the 2.4 m space echelle for the others (scripts/tc_limits.py,
host_class). The mixed mass cancels in
Delta K = (log eps_limit - log eps_threshold) / 10, so a negative Delta K says that the spectrum excludes the beacon
that the same star would show at threshold, whatever the mixed mass.

Referee comment 2.5: the archival limits are a per-pixel 3-sigma criterion, and the detections a matched filter.
scripts/tc_detect.py gives, per spectrum, the abundance le_clean that raises its clean-line matched-filter statistic
by z = 5, and the statistic z0 of the spectrum without Tc. For weak lines the statistic is linear in the abundance,
so the 3-sigma upper limit in the same statistic is le_clean + log10((max(z0, 0) + 3) / 5).

    uv run python scripts/own_thresholds.py [key ...]     # writes research/tc_limits/own_thresholds.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

from beacon import detect, hosts, survey  # noqa: E402

INST = {i.name: i for i in detect.INSTRUMENTS}
GROUND, SPACE = INST["8.2 m, R = 140,000"], INST["2.4 m space UV, R = 114,000"]


def threshold(key: str, star, vsini: float) -> tuple[str, float]:
    s = T.synths(star)[1.0]
    cool = T.host_class(key, star) == "FGK"
    sp, inst = ("Tc I", GROUND) if cool else ("Tc II", SPACE)
    lines = [ln for ln in survey.SPECIES[sp]() if inst.wmin <= ln.wl <= inst.wmax]
    h = hosts.Host("x", star.name, star.teff, star.logg, 1.0, 1.0, 1.0, 1.0, 1.0, vsini, 3.5 if cool else 3.0)
    ew = np.array([[s.ew(ln, le) for le in survey.LOG_EPS_GRID] for ln in lines])
    fw = np.full(len(lines), survey.intrinsic_fwhm(h, 98.906))
    c = survey.Curves(h, sp, lines, ew, fw, np.ones(len(lines)))
    return sp, survey.required_log_eps(c, inst, 300.0)


def main():
    arch = {r["star"]: r for r in csv.DictReader((T.ROOT / "paper" / "data" / "archival_limits.csv").open(encoding="utf-8"))}
    det = {}
    for r in csv.DictReader((T.OUT / "detect.csv").open()):
        if not r["le_clean"] or not r["z0"]:
            continue
        le_mf = float(r["le_clean"]) + np.log10((max(float(r["z0"]), 0.0) + 3.0) / 5.0)
        if r["key"] not in det or le_mf < det[r["key"]][0]:
            det[r["key"]] = (le_mf, r["instrument"], r["limit"], float(r["le_clean"]), float(r["z0"]))
    # with star keys as arguments, only those are recomputed and the others kept from the last run
    only = set(sys.argv[1:])
    path = T.OUT / "own_thresholds.csv"
    old = {r["key"]: r for r in csv.DictReader(path.open(encoding="utf-8"))} if only and path.exists() else {}
    rows = []
    for r in csv.DictReader((T.OUT / "summary.csv").open(encoding="utf-8")):
        key = r["key"]
        if not r["log_eps"] or key not in T.STARS:
            continue
        star = T.STARS[key]
        a = arch.get(star.name, {})
        vsini = float(a.get("vsini_kms") or 0.0) if a.get("vsini_kms") not in (None, "") else 0.0
        if only and key not in only:
            rows.append(old[key])
            continue
        sp, thr = threshold(key, star, vsini)
        lim = float(r["log_eps"])
        mf = det.get(key)
        rows.append(dict(key=key, name=star.name, type=a.get("type", ""), teff=star.teff, logg=star.logg,
                         grid=T.grid_name(star), vsini=vsini, species=sp, threshold_own=round(thr, 2),
                         limit_pixel=lim, dK_own=round((lim - thr) / 10, 3),
                         limit_mf3=round(mf[0], 2) if mf else "", mf_instrument=mf[1] if mf else ""))
        print(f"{star.name:28s} {a.get('type', ''):10s} {star.teff:6.0f} {sp:6s} threshold {thr:+.2f}  limit {lim:+.1f}  "
              f"dK_own {(lim - thr) / 10:+.3f}  MF 3-sigma {mf[0] if mf else float('nan'):+.2f}", flush=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
