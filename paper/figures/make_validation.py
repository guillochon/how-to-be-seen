"""Solar Fe I validation: model versus Kitt Peak atlas equivalent widths. Writes paper/figures/validation.pdf.

    uv run python scripts/validate_sun.py          # makes research/validation_sun.csv
    uv run python paper/figures/make_validation.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader((ROOT / "research" / "validation_sun.csv").open()))
wl = np.array([float(r["wl_A"]) for r in rows])
mod = np.array([float(r["ew_model_mA"]) for r in rows])
obs = np.array([float(r["ew_atlas_mA"]) for r in rows])
chi = np.array([float(r["elow_eV"]) for r in rows])
d = np.log10(obs / mod)

style.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(style.COLUMN, 1.15 * style.COLUMN),
                             gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.08))
sc = a1.scatter(mod, obs, c=chi, s=7, cmap="viridis", vmin=0, vmax=5, lw=0)
g = np.array([3, 150])
a1.plot(g, g, color=style.MUTED, lw=0.8)
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlim(3, 150)
a1.set_ylim(3, 150)
a1.set_ylabel(r"$W_\mathrm{atlas}$ (mÅ)")
a1.tick_params(labelbottom=False)
cb = fig.colorbar(sc, ax=[a1, a2], pad=0.02, shrink=0.6, anchor=(0, 1))
cb.set_label(r"$\chi_\mathrm{low}$ (eV)")
a1.text(0.04, 0.92, f"{len(rows)} Fe I lines\nmean offset {d.mean():+.2f} dex\nscatter {d.std():.2f} dex",
        transform=a1.transAxes, va="top", fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)

a2.scatter(mod, d, c=chi, s=7, cmap="viridis", vmin=0, vmax=5, lw=0)
a2.axhline(0, color=style.MUTED, lw=0.8)
a2.axhline(d.mean(), color=style.PALETTE[1], lw=0.8, ls="--")
# mean residual by excitation energy: the Tc I resonance lines are low-excitation lines
cmap = plt.get_cmap("viridis")
txt = []
for lo, hi, lab in ((0.0, 3.0, "< 3"), (3.0, 4.0, "3$-$4"), (4.0, 9.0, r"$\geq$ 4")):
    m = d[(chi >= lo) & (chi < hi)].mean()
    a2.plot([3.3, 5.0], [m, m], color=cmap(min((lo + hi) / 2, 5.0) / 5.0), lw=1.6, solid_capstyle="butt")
    txt.append(f"{lab} eV: {m:+.2f}")
a2.text(0.97, 0.05, r"mean by $\chi_\mathrm{low}$" + "\n" + "\n".join(txt), transform=a2.transAxes, ha="right",
        va="bottom", fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)
a2.set_xscale("log")
a2.set_xlim(3, 150)
a2.set_ylim(-0.35, 0.35)
a2.set_xlabel(r"$W_\mathrm{model}$ (mÅ)")
a2.set_ylabel(r"$\log W_\mathrm{atlas}/W_\mathrm{model}$")
style.save(fig, "validation")
