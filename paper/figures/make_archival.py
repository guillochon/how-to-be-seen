"""Upper limits on the Kardashev type of a Tc beacon in each star with archival spectra, against the threshold of
its host class, drawn across the temperatures of that class. Rings mark the stars of Table 5.
Writes paper/figures/archival.pdf.

Reads research/tc_limits/summary.csv (scripts/tc_batch.py --summary) and the thresholds in
research/tc_limits/archival_stats.json (scripts/archival_table.py). Run that script first.

    uv run python paper/figures/make_archival.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import style

ROOT = Path(__file__).resolve().parents[2]
rows = [r for r in csv.DictReader((ROOT / "research" / "tc_limits" / "summary.csv").open(encoding="utf-8"))
        if r["log_eps"] and r["K_host"]]
th = json.loads((ROOT / "research" / "tc_limits" / "archival_stats.json").read_text())["thresholds"]
HUMANITY_K = 0.72

classes = [("quiet late-B", "Quiet late-B (atmosphere)", style.PALETTE[0], "o"),
           ("A star", "Am and Ap stars (turbulent layer)", style.PALETTE[2], "D"),
           ("HB", "Horizontal branch (turbulent layer)", style.PALETTE[3], "^"),
           ("FGK", "FGK stars (convection zone)", style.PALETTE[1], "s")]
TOP = 5  # the stars of each class listed in the paper's table

style.apply()
fig, ax = plt.subplots(figsize=(style.COLUMN, 0.8 * style.COLUMN))
for host, label, col, mk in classes:
    sel = [r for r in rows if r["host"] == host]
    t = [float(r["teff"]) for r in sel]
    ax.scatter(t, [float(r["K_host"]) for r in sel], s=7, marker=mk, color=col,
               edgecolors="none", alpha=0.85, label=label, zorder=3)
    # each class's threshold, drawn only across the temperatures of its own stars
    ax.plot([min(t), max(t)], [th[host]["K"]] * 2, color=col, lw=1.0, ls="--", zorder=2)
    top = sorted(sel, key=lambda r: float(r["K_host"]))[:TOP]
    ax.scatter([float(r["teff"]) for r in top], [float(r["K_host"]) for r in top], s=22, marker="o",
               facecolors="none", edgecolors=style.INK, linewidths=0.5, zorder=4)
ax.axhline(HUMANITY_K, color=style.MUTED, lw=0.8, ls=":", zorder=1)
ax.text(0.98, HUMANITY_K - 0.012, "humanity today", transform=ax.get_yaxis_transform(), ha="right", va="top",
        fontsize=style.LABEL_SIZE, color=style.MUTED, bbox=style.LABEL_BOX)
ax.set_ylim(0.55, 2.0)  # room above the data for the legend
ax.set_xscale("log")
ax.set_xlim(18000, 5400)
ax.set_xticks([6000, 8000, 10000, 14000])
ax.set_xticklabels(["6000", "8000", "10000", "14000"])
ax.minorticks_off()
ax.set_xlabel(r"$T_\mathrm{eff}$ (K)")
ax.set_ylabel(r"$K$ of the largest beacon allowed")
ax.legend(loc="upper left", **style.LEGEND)
style.save(fig, "archival")
