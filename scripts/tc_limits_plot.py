"""Plots of the Tc limits of scripts/tc_limits.py: the stacked spectra around the Tc lines that set the
tightest limits, with the Tc-only profile at the limit and at LE_REF, and the limit from every line.

    uv run python scripts/tc_limits_plot.py hd101065     # writes research/tc_limits/<star>/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

from beacon import atlas, atoms, synth  # noqa: E402

N_PANEL = 12
LE_REF = 2.5  # upper limit of Andrievsky et al. (2023) for HD 101065


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("star", choices=sorted(T.STARS))
    key = ap.parse_args().star
    out = T.OUT / key
    res = json.loads((T.OUT / f"{key}.json").read_text())
    st = np.load(out / "stacks.npz")
    dil = T.STARS[key].dilution  # a companion's light makes the lines shallower
    s = synth.Synth(atlas.model(res["model"]["teff"], res["model"]["logg"]), vturb=1.0)
    rows = [r for r in res["tc_lines"] if r["limit_b1"] is not None and np.isfinite(r["limit_b1"])]
    if not rows:  # no Tc line sets a limit (HD 23631: ESPaDOnS spectra at S/N 6)
        print(f"{key}: no Tc line sets a limit, nothing to plot")
        return
    rows.sort(key=lambda r: r["limit_b1"])
    picks = rows[:N_PANEL]
    for name in ("4297", "4262", "4238", "3237", "2543"):
        extra = [r for r in rows if f"{r['wl']:.0f}" == name and r not in picks]
        picks += extra[:1]
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, picks, strict=False):
        inst = r["instrument"]
        w, rr = st[f"{inst}_w"], st[f"{inst}_r"]
        ion = 0 if r["species"] == "Tc I" else 1
        ln = next(x for x in atoms.kurucz_lines("gf4300.pos" if ion == 0 else "gf4301.pos", 43, ion, r["wl"] - 0.01,
                                                r["wl"] + 0.01) if abs(x.wl - r["wl"]) < 1e-3)
        prof = T.Profiles(s, ln, r["fwhm_kms"])
        hw = 4 * r["fwhm_kms"]
        vo = (w / ln.wl - 1) * 2.998e5
        sel = np.abs(vo) < hw
        ax.plot(vo[sel], rr[sel], color="0.2", lw=0.8, drawstyle="steps-mid")
        for le, col, lab in ((r["limit_b1"], "C3", f"limit {r['limit_b1']:.1f}"), (LE_REF, "C0", f"{LE_REF}")):
            ax.plot(prof.v, 1 - dil * (1 - prof(le)), color=col, lw=1.2, label=lab)
        ax.axvspan(-0.5 * r["fwhm_kms"], 0.5 * r["fwhm_kms"], color="0.9", zorder=0)
        ax.set_xlim(-hw, hw)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{r['species']} {r['wl']:.2f} ({inst.upper()})", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", frameon=False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    fig.supxlabel("velocity from line centre (km/s)")
    fig.supylabel("flux / upper envelope")
    fig.tight_layout()
    fig.savefig(out / "tc_lines.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for b, mk in zip((f"{b:g}" for b in T.BLANKET), "os^v", strict=False):
        for inst, col in (("harps", "C0"), ("uves", "C1"), ("iue", "C2")):
            pts = [(r["wl"], r[f"limit_b{b}"]) for r in res["tc_lines"]
                   if r["instrument"] == inst and r[f"limit_b{b}"] is not None and np.isfinite(r[f"limit_b{b}"])]
            if pts:
                x, y = np.array(pts).T
                ax.scatter(x, y, s=12, marker=mk, color=col, alpha=0.8 if b == "1" else 0.35,
                           label=f"{inst.upper()}, opacity x{b}")
    fe = [r["limit_b1"] for r in res["fe_lines"] if r["limit_b1"] is not None]
    ax.axhline(LE_REF, color="0.5", ls="--", lw=1)
    ax.text(1900, LE_REF + 0.08, "Andrievsky et al. (2023)", fontsize=8, color="0.4")
    ax.set_xlabel("wavelength (A)")
    ax.set_ylabel("upper limit on log eps(Tc)")
    ax.set_title(f"{res['star']['name']}: limit from each Tc line"
                 + (f" (lowest of {len(fe)} Fe-line limits: {min(fe):.1f})" if fe else ""),
                 fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "limits.png", dpi=150)
    print("wrote", out / "tc_lines.png", out / "limits.png")


if __name__ == "__main__":
    main()
