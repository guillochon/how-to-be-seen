"""Write the table that compares photospheric and gas beacons of bred 99Tc.

Reads research/results.json, research/comet.json, research/comet_opt.json and research/broadcast.json,
so run scripts/results.py, comet.py, comet_opt.py and broadcast.py first.

    uv run python scripts/beacon_table.py     # writes paper/tables/beacons.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUND = "8.2 m, R = 140,000"
SPACE = "2.4 m space UV, R = 114,000"
RSUN, AU = 6.957e10, 1.495978707e13  # cm
TC1, TC2 =r"Tc\,\textsc{i}", r"Tc\,\textsc{ii}"


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def comet_sky(q: float, r_sub: float) -> float:
    """Fraction of the sky from which a parabolic comet of periastron q, releasing its dopant inside r_sub (both in
    stellar radii), passes in front of its star: at true anomaly phi, observers within R*/r(phi) of the orbital
    plane see it transit, so the fraction is (1 / 2 pi) int R*/r dphi over the release arc, r = 2 q / (1 + cos phi)."""
    phi = np.arccos(2 * q / r_sub - 1)
    return float((2 * phi + 2 * np.sin(phi)) / (2 * q) / (2 * np.pi))


def load(name: str) -> dict:
    return json.loads((ROOT / "research" / f"{name}.json").read_text())


def main():
    res, com, opt, bro = (load(n) for n in ("results", "comet", "comet_opt", "broadcast"))
    sp = res["hosts"]
    sun = sp["G"]["species"]["Tc I"][f"{GROUND}|300"]
    am = sp["Am"]["species"]["Tc II"][f"{SPACE}|300"]
    lateb = sp["B"]["species"]["Tc II"][f"{SPACE}|300"]
    nat = com["design"]["rows"]["beta Pic-like|Tc II"]
    tuned = com["design"]["rows"]["timed, narrow|Tc II"]
    best = opt["best"]
    best_k = opt["sensitivity"]["100|Mn"]["sagan_K"]
    belt = next(r for r in bro["rows"] if r["theta_deg"] == 3)
    shell = next(r for r in bro["rows"] if r["theta_deg"] == 90)
    sky_c = comet_sky(com["design"]["q_rstar"], com["design"]["rsub_rstar"])
    rows = [
        ("Photosphere", "Sun", f"{TC1}, 8.2 m", "1", sun["mass_g"] / 1e3, sun["sagan_K_best"]),
        ("Photosphere", "Am star", f"{TC2}, 2.4 m space", "1", am["mass_g"] / 1e3, am["sagan_K_best"]),
        ("Photosphere", "Late B", f"{TC2}, 2.4 m space", "1", lateb["mass_g"] / 1e3, lateb["sagan_K_best"]),
        (r"Comet, $\beta$~Pic-like", "Am star", f"{TC2}, 2.4 m space", f"{sky_c:.3f}",
         nat["mass_event_g"] / 1e3, nat["sagan_K"]),
        ("Comet, timed release", "Am star", f"{TC2}, 2.4 m space", f"{sky_c:.3f}",
         tuned["mass_event_g"] / 1e3, tuned["sagan_K"]),
        ("Artificial comet", "Sun", f"{TC1}, 3.6 m", sci(0.25 * (RSUN / (best["r_au"] * AU)) ** 2),
         best["m_event_g"] / 1e3, best_k),
        (r"Belt, $|b| < 3^\circ$", "Sun", f"{TC1}, 3.6 m", f"{belt['sky_fraction']:.3f}", belt["mass_kg"],
         belt["sagan_K"]),
        ("Shell", "Sun", f"{TC1}, 3.6 m", "1", shell["mass_kg"], shell["sagan_K"]),
    ]
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Beacon & Host & Lines, telescope & Sky & Mass (kg) & $K$ \\", r"\midrule"]
    for i, (name, host, tel, sky, mass, k) in enumerate(rows):
        lines.append(f"{name} & {host} & {tel} & {sky} & {sci(mass)} & {k:.2f} \\\\")
        if i in (2, 4):
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "beacons.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/tables/beacons.tex")


if __name__ == "__main__":
    main()
