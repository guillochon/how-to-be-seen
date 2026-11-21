"""The Tc and Pm lines of the paper, for the appendix table and the machine-readable line list.

Threshold lines: the three Tc I resonance lines, the Tc II lines with log gf >= -1 and the Pm II lines that enter
the detection statistic of Section 3.4 (beacon/survey.py). Archival lines: every Tc line that the archival limits of
Section 6 used in some star (research/tc_limits/*.json), flagged where it set a star's limit
(paper/data/archival_limits.csv).

    uv run python scripts/line_table.py    # writes paper/tables/lines.tex and paper/data/tc_lines.csv
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beacon import atoms  # noqa: E402

CM_PER_EV = 8065.544


def main():
    limit_set = set()
    with (ROOT / "paper" / "data" / "archival_limits.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sp, wl = r["line"].rsplit(" ", 1)
            limit_set.add((sp, round(float(wl), 2)))
    used = {}
    for f in sorted((ROOT / "research" / "tc_limits").glob("*.json")):
        for ln in json.loads(f.read_text()).get("tc_lines", []):
            key = (ln["species"], round(ln["wl"], 2))
            used.setdefault(key, (ln["loggf"], ln["elow"], set()))[2].add(f.stem)

    thresh = {"Tc I": atoms.tc1_resonance(), "Tc II": atoms.tc2_lines(min_loggf=-1.0), "Pm II": atoms.pm2_lines()}
    cols = {}
    for sp, lines in thresh.items():
        rows = {round(ln.wl, 2): [ln.loggf, ln.elow / CM_PER_EV, "t"] for ln in lines}
        for (s, wl), (gf, el, _stars) in used.items():
            if s == sp and (s, wl) in limit_set:
                rows.setdefault(wl, [gf, el / CM_PER_EV, ""])[2] += "a"
        cols[sp] = sorted(rows.items())

    head = r"$\lambda$ (\AA) & $\chi$ (eV) & $\log gf$ & "
    out = [r"\begin{tabular}{rrrl@{\hspace{10pt}}rrrl@{\hspace{10pt}}rrrl}", r"\toprule",
           r"\multicolumn{4}{c}{Tc\,\textsc{i}} & \multicolumn{4}{c}{Tc\,\textsc{ii}} & \multicolumn{4}{c}{Pm\,\textsc{ii}} \\",
           r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
           " & ".join([head + "Use"] * 3) + r" \\", r"\midrule"]
    for trio in zip_longest(cols["Tc I"], cols["Tc II"], cols["Pm II"]):
        cells = []
        for item in trio:
            if item is None:
                cells.append(" & & & ")
            else:
                wl, (gf, chi, use) = item
                cells.append(f"{wl:.2f} & {chi:.2f} & ${gf:+.3f}$ & {use}")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "lines.tex").write_text("\n".join(out) + "\n")

    with (ROOT / "paper" / "data" / "tc_lines.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "wl_air_A", "elow_eV", "loggf", "in_threshold_statistic", "sets_an_archival_limit",
                    "n_stars_used"])
        tset = {(sp, round(ln.wl, 2)) for sp in ("Tc I", "Tc II") for ln in thresh[sp]}
        for (sp, wl), (gf, el, stars) in sorted(used.items()):
            w.writerow([sp, f"{wl:.2f}", f"{el / CM_PER_EV:.3f}", f"{gf:.3f}", int((sp, wl) in tset),
                        int((sp, wl) in limit_set), len(stars)])
    print(f"wrote paper/tables/lines.tex ({max(len(v) for v in cols.values())} rows) and paper/data/tc_lines.csv "
          f"({len(used)} Tc lines)")


if __name__ == "__main__":
    main()
