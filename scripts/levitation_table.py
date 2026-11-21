"""Write the table of the radiative acceleration on Tc against gravity in A and late-B atmospheres.

Reads research/levitation.json, so run scripts/levitation.py first.

    uv run python scripts/levitation_table.py     # writes paper/tables/levitation.tex
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGG = 4.0


def weakest(rec: dict, key: str = "log_grad_diffw") -> float:
    """log(g_rad / g) at the depth where the support is weakest, tau5000 = 1e-4 to 3."""
    return min(rec[key]) - LOGG


def main():
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    teffs = sorted(int(k.split("|")[1]) for k in lev if k.startswith("threshold|"))
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$T_\mathrm{eff}$ (K) & $\log\epsilon_\mathrm{Tc}$ & Unsaturated & Threshold & $100\times$ threshold \\",
             r"\midrule"]
    for t in teffs:
        lo, hi = lev[f"tc|{t}|+0"], lev[f"tc|{t}|+2"]
        lines.append(f"{t} & ${lev[f'threshold|{t}']:+.2f}$ & {weakest(lo, 'log_grad_unsat_diffw'):.2f} & "
                     f"{weakest(lo):.2f} & {weakest(hi):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "levitation.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/tables/levitation.tex")


if __name__ == "__main__":
    main()
