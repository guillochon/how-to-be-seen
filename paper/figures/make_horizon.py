"""Isotope inventory needed for a 5-sigma detection against distance. Writes paper/figures/horizon.pdf.

Photon-limited S/N per pixel for a 10 h exposure (beacon/detect.py), capped at S/N = 1000 to
represent the systematic floor of template subtraction. Reads research/curves.pkl.

    uv run python paper/figures/make_horizon.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from beacon import detect, hosts, survey  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

T_EXP = 10 * 3600.0
CAP = 1000.0
curves = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))
H = {h.key: h for h in hosts.host_grid()}
INST = {i.name: i for i in detect.INSTRUMENTS}
COLORS = {
    "3.6 m, R = 115,000": style.PALETTE[2],
    "8.2 m, R = 140,000": style.PALETTE[0],
    "39 m, R = 100,000": style.PALETTE[1],
    "2.4 m space UV, R = 114,000": style.PALETTE[3],
    "6 m space UV-optical, R = 100,000": style.PALETTE[4],
}
d_pc = np.logspace(0, np.log10(3000), 30)

panels = [
    ("G", "Sun-like host (5777 K)", [("Tc I", "-"), ("Pm II", "--")],
     ["3.6 m, R = 115,000", "8.2 m, R = 140,000", "39 m, R = 100,000", "6 m space UV-optical, R = 100,000"]),
    ("Am", "Am-star host (8000 K)", [("Pm II", "--"), ("Tc II", ":")],
     ["3.6 m, R = 115,000", "8.2 m, R = 140,000", "39 m, R = 100,000", "2.4 m space UV, R = 114,000",
      "6 m space UV-optical, R = 100,000"]),
]

style.apply()
fig, axes = plt.subplots(2, 1, figsize=(style.COLUMN, 1.45 * style.COLUMN), sharex=True,
                         gridspec_kw=dict(hspace=0.12))
LABEL = {"Tc I": r"$^{99}$Tc, Tc I", "Pm II": r"$^{147}$Pm, Pm II", "Tc II": r"$^{99}$Tc, Tc II (UV)"}
for ax, (hk, title, species, insts) in zip(axes, panels, strict=True):
    h = H[hk]
    drawn_inst, drawn_sp = [], []
    MV = survey.absolute_v(h.teff)
    for sp, ls in species:
        c = curves[hk][sp]
        nuc = "147Pm" if sp.startswith("Pm") else "99Tc"
        for name in insts:
            inst = INST[name]
            if not any(inst.wmin <= ln.wl <= inst.wmax for ln in c.lines):
                continue
            m = []
            for d in d_pc:
                V = MV + 5 * np.log10(d / 10.0)
                le = survey.required_log_eps(c, inst, (V, T_EXP, CAP))
                m.append(hosts.isotope_mass(le, hosts.NUCLIDES[nuc]["A"], h.mmix) / 1e3 if np.isfinite(le) else np.nan)
            ax.plot(d_pc, m, color=COLORS[name], ls=ls, lw=1.0)
            if np.isfinite(m).any():
                drawn_inst += [name] * (name not in drawn_inst)
                drawn_sp += [(sp, ls)] * ((sp, ls) not in drawn_sp)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Isotope inventory (kg)")
    ax.text(0.03, 0.93, title, transform=ax.transAxes, va="top", fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)
    # legends list only what is drawn in the panel: instruments by color, species by line style
    h1 = [Line2D([], [], color=COLORS[n], lw=1.0, label=n) for n in COLORS if n in drawn_inst]
    ax.add_artist(ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 0.88),
                            **{**style.LEGEND, "fontsize": 5.5}))
    h2 = [Line2D([], [], color=style.INK, ls=ls, label=LABEL[sp]) for sp, ls in drawn_sp]
    ax.legend(handles=h2, loc="lower right", **style.LEGEND)
axes[-1].set_xlabel("Distance (pc)")
axes[-1].set_xlim(1, 3000)
style.save(fig, "horizon")
