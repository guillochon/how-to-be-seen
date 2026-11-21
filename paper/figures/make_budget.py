"""Isotope inventory and fission energy that a detectable beacon needs, by host star.
Writes paper/figures/budget.pdf.

Reads research/results.json (scripts/results.py). Ground: 8.2 m, R = 140,000. Space: 2.4 m UV
echelle, R = 114,000 (Tc II, A stars only). Error bars span the mixed-mass range of each host.

    uv run python paper/figures/make_budget.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

ROOT = Path(__file__).resolve().parents[2]
res = json.loads((ROOT / "research" / "results.json").read_text())
H = res["hosts"]
order = ["M", "K", "G", "F", "Am", "A0", "B"]
xl = ["M\n3500 K", "K\n4500 K", "Sun\n5777 K", "F\n6250 K", "Am\n8000 K", "A0\n9500 K", "late B\n12250 K"]
GROUND = "8.2 m, R = 140,000"
SPACE = "2.4 m space UV, R = 114,000"
UVBLUE = "8.2 m UV-blue, R = 80,000"
series = [
    ("Tc I", GROUND, r"$^{99}$Tc, Tc I (ground)", style.PALETTE[0], "o"),
    ("Tc II", UVBLUE, r"$^{99}$Tc, Tc II (ground near-UV)", style.PALETTE[3], "^"),
    ("Tc II", SPACE, r"$^{99}$Tc, Tc II (space UV)", style.PALETTE[2], "D"),
    ("Pm II", GROUND, r"$^{147}$Pm, Pm II (ground)", style.PALETTE[1], "s"),
]

style.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(style.COLUMN, 1.12 * style.COLUMN), sharex=True,
                             gridspec_kw=dict(hspace=0.08))
x = np.arange(len(order))
for j, (sp, inst, lab, col, mk) in enumerate(series):
    off = (j - 1.5) * 0.16
    for snr, fill in ((300, True), (1000, False)):
        xs, m, lo, hi, e = [], [], [], [], []
        for i, hk in enumerate(order):
            r = H[hk]["species"].get(sp, {}).get(f"{inst}|{snr}")
            if r is None:
                continue
            xs.append(i + off)
            m.append(r["mass_g"] / 1e3)
            lo.append(r["mass_lo_g"] / 1e3)
            hi.append(r["mass_hi_g"] / 1e3)
            e.append(r["upkeep_best_W"])
        m, lo, hi = np.array(m), np.array(lo), np.array(hi)
        kw = dict(color=col, marker=mk, ms=4, ls="none", mfc=col if fill else "white", mec=col, mew=0.8)
        a1.errorbar(xs, m, yerr=[m - lo, hi - m], elinewidth=0.6, capsize=0,
                    label=lab if fill else None, **kw)
        a2.plot(xs, e, **kw)
a1.set_yscale("log")
a1.set_ylabel("Isotope inventory (kg)")
mearth = 5.9722e24
sec = a1.secondary_yaxis("right", functions=(lambda v: v / mearth, lambda v: v * mearth))
sec.set_ylabel(r"($M_\oplus$)")
from matplotlib.lines import Line2D  # noqa: E402

snr_keys = [Line2D([], [], color=style.INK, marker="o", ms=4, ls="none", label="filled: S/N = 300"),
            Line2D([], [], color=style.INK, marker="o", ms=4, ls="none", mfc="white", label="open: S/N = 1000")]
a1.legend(handles=a1.get_legend_handles_labels()[0] + snr_keys, loc="lower left", **style.LEGEND)
a2.text(0.03, 0.05, "filled: S/N = 300\nopen: S/N = 1000", transform=a2.transAxes, ha="left", va="bottom",
        fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)
a2.set_yscale("log")
a2.set_ylabel("Upkeep power (W)")
sec2 = a2.secondary_yaxis("right", functions=(lambda p: (np.log10(np.maximum(p, 1e-30)) - 6) / 10,
                                              lambda k: 10 ** (10 * k + 6)))
sec2.set_ylabel(r"Sagan type $K$")
sec2.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], labels=["1.0", "1.2", "1.4", "1.6", "1.8", "2.0"])
sec2.minorticks_off()
a2.axhline(1.7e17, color=style.MUTED, lw=0.6, ls="--")
a2.text(-0.4, 2.3e17, "Earth's insolation", fontsize=style.LABEL_SIZE, color=style.MUTED, va="bottom")
a2.set_xticks(x)
a2.set_xticklabels(xl)
a2.set_xlim(-0.5, len(order) - 0.5)
style.save(fig, "budget")
