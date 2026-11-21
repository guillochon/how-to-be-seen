"""Write the LaTeX tables of the paper from research/results.json.

    uv run python scripts/make_tables.py     # writes paper/tables/hosts.tex and paper/tables/budget.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beacon.const import AMU, YR  # noqa: E402

GROUND = "8.2 m, R = 140,000"
SPACE = "2.4 m space UV, R = 114,000"
UVBLUE = "8.2 m UV-blue, R = 80,000"
ORDER = ["M", "K", "G", "F", "Am", "A0", "B"]
NAMES = {"M": "M dwarf", "K": "K dwarf", "G": "Sun", "F": "F dwarf", "Am": "Am star", "A0": "A0 star", "B": "Late B"}


def sci(x: float, digits: int = 1) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.{digits}f}\times10^{{{e}}}$"


# Natural abundance of 235U, and mass (u) and natural abundance of the capture targets 98Mo and 144Sm (IUPAC).
U235_FRAC = 0.007204
TARGET = {"99Tc capture": (98.906, 97.905, 0.2439), "145Pm capture": (144.913, 143.912, 0.0307)}


def _atoms(r: dict) -> float:
    route = r["best_route"]
    return r["routes"][route]["mass_g"] / (TARGET[route][0] * AMU)


def fuel_kg(r: dict) -> float:
    """235U fissioned to breed the inventory once, one capture per atom, 1.46 spare neutrons per fission."""
    return _atoms(r) / 1.46 * 235.044 * AMU / 1e3


def target_kg(r: dict, sp: str) -> float:
    """Natural Mo (Sm) that holds the 98Mo (144Sm) targets for one filling."""
    _, a, f = TARGET[r["best_route"]]
    return _atoms(r) * a * AMU / f / 1e3


def main():
    res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    lines = [r"\begin{tabular}{lrrrrc}", r"\toprule",
             r"Host & $T_\mathrm{eff}$ (K) & $\log g$ & $M_\star$ ($M_\odot$) & $M_V$ & $\log(M_\mathrm{mix}/M_\odot)$ [range] \\",
             r"\midrule"]
    for k in ORDER:
        h = res[k]
        lm = [np.log10(h[x]) for x in ("mmix", "mmix_lo", "mmix_hi")]
        rng = f"${lm[0]:.2f}$ [${lm[1]:.2f}$, ${lm[2]:.2f}$]" if lm[1] != lm[2] else f"${lm[0]:.2f}$"  # no range: late-B
        lines.append(f"{NAMES[k]} & {h['teff']:.0f} & {h['logg']:.2f} & {h['mass']:.2f} & {h['MV']:.2f} & {rng} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "hosts.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
             r"Host & Lines & Telescope & $\log\epsilon$ & Inventory (kg) & $K_\mathrm{fis}$ & $P_\mathrm{up}$ (W) & $K$ & "
             r"$^{235}$U (kg) & Target (kg) \\",
             r"\midrule"]
    rows = (("Tc I", GROUND, r"Tc\,\textsc{i}", "8 m"),
            ("Tc II", UVBLUE, r"Tc\,\textsc{ii}", "8 m, UV"),
            ("Tc II", SPACE, r"Tc\,\textsc{ii}", "2.4 m, space"),
            ("Pm II", GROUND, r"Pm\,\textsc{ii}", "8 m"))
    for k in ORDER:
        h = res[k]
        first = True
        for sp, inst, lab, tel in rows:
            r = h["species"].get(sp, {}).get(f"{inst}|300")
            if r is None:
                continue
            lines.append(f"{NAMES[k] if first else ''} & {lab} & {tel} & ${r['log_eps']:+.2f}$ & {sci(r['mass_g'] / 1e3)} & "
                         f"{r['sagan_K']:.2f} & {sci(r['upkeep_best_W'])} & {r['sagan_K_best']:.2f} & "
                         f"{sci(fuel_kg(r))} & {sci(target_kg(r, sp))} \\\\")
            first = False
        if k != ORDER[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "budget.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/tables/hosts.tex, paper/tables/budget.tex")
    _ = YR


if __name__ == "__main__":
    main()
