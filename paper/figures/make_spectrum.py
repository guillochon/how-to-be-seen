"""The beacon in a real spectrum: Tc II lines injected into the HST STIS E230H spectrum of the HgMn star HD 175640.
Writes paper/figures/spectrum.pdf.

The stack is the rest-frame, normalized E230H spectrum of scripts/tc_limits.py (research/tc_limits/hd175640).
The Tc II profiles are synthesized in the star's own ATLAS9 model and broadened to the stack's line width
(scripts/tc_limits.py Profiles). Abundances are the 5-sigma threshold of the late-B host read from space
(research/results.json, 2.4 m, R = 114,000, S/N = 300) and ten times that. A line is clean when the stack stays
above 0.9 of the local continuum within +/- FWHM/2 of its centre (scripts/tc_detect.py).

    uv run python paper/figures/make_spectrum.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tc_detect as D  # noqa: E402
import tc_limits as T  # noqa: E402

from beacon import atoms  # noqa: E402

KEY, INST = "hd175640", "e230h"
CKMS = 2.998e5
res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]["B"]["species"]["Tc II"]
le_thr = res["2.4 m space UV, R = 114,000|300"]["log_eps"]

star = T.STARS[KEY]
b = json.loads((T.OUT / f"{KEY}.json").read_text())
syn = T.synths(star)[1.0]
d = np.load(T.OUT / KEY / "stacks.npz")
w, r, sig = d[INST + "_w"], d[INST + "_r"], d[INST + "_sig"]
ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
w, r, sig = w[ok], r[ok], sig[ok]
recs = [ln for ln in b["tc_lines"] if ln["instrument"] == INST and ln["species"] == "Tc II"]
fwhm = recs[0]["fwhm_kms"]
lines = [atoms.Line(ln["wl"], ln["loggf"], ln["elow"], 43, 1) for ln in recs]
prof = {ln.wl: T.Profiles(syn, ln, fwhm) for ln in lines}


def with_tc(le):
    rr = r.copy()
    for ln in lines:
        p = prof[ln.wl]
        v = (w / ln.wl - 1) * CKMS
        sel = np.abs(v) < p.v[-1]
        rr[sel] *= np.interp(v[sel], p.v, p(le))
    return rr


y1, y10 = with_tc(le_thr), with_tc(le_thr + 1)
panels = [2543.223, 2609.996, 2647.011]

style.apply()
fig = plt.figure(figsize=(style.PAGE, 0.36 * style.PAGE))
outer = fig.add_gridspec(1, len(panels), wspace=0.3)
for k, wl in enumerate(panels):
    inner = outer[k].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
    a = fig.add_subplot(inner[0])
    bx = fig.add_subplot(inner[1], sharex=a)
    sel = np.abs(w - wl) < 0.6
    x = w[sel]
    cont = D.local_cont(w, r, wl)
    clean = D.is_clean(w, r, wl, fwhm)
    a.plot(x, r[sel] / cont, color=style.INK, lw=0.8, drawstyle="steps-mid", label="HD 175640")
    a.plot(x, y10[sel] / cont, color=style.PALETTE[1], lw=0.8, drawstyle="steps-mid", label=r"$+$Tc, $10\times$ thr.")
    s = sig[sel] / cont
    bx.fill_between(x, -100 * s, 100 * s, color=style.MUTED, alpha=0.25, lw=0, step="mid", label=r"$\pm1\sigma$ per pixel")
    bx.plot(x, 100 * (y1 - r)[sel] / cont, color=style.PALETTE[0], lw=0.8, drawstyle="steps-mid", label=r"$5\sigma$ threshold")
    bx.plot(x, 100 * (y10 - r)[sel] / cont, color=style.PALETTE[1], lw=0.8, drawstyle="steps-mid", label=r"$10\times$ threshold")
    for ax in (a, bx):
        ax.axvline(wl, color=style.MUTED, lw=0.5, ls=":")
    a.set_xlim(x[0], x[-1])
    a.set_ylim(0.0, 1.05)
    a.tick_params(labelbottom=False)
    a.set_title(rf"Tc II {wl:.0f} ({'clean' if clean else 'blended'})", fontsize=style.FONT_SIZE, pad=2)
    if k == 0:
        a.set_ylabel("Flux")
        bx.set_ylabel(r"$\Delta F$ (%)")
    lo = min(-100 * s.max(), 100 * ((y10 - r)[sel] / cont).min()) * 1.2
    bx.set_ylim(lo, -0.35 * lo)
    bx.xaxis.set_major_locator(MaxNLocator(4))
    bx.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    bx.set_xlabel("Air wavelength (Å)")
    if k == len(panels) - 1:
        a.legend(loc="lower left", bbox_to_anchor=(0.02, 0.05), **style.LEGEND)
style.save(fig, "spectrum")
print(f"threshold log eps {le_thr:.2f}; clean:", {wl: D.is_clean(w, r, wl, fwhm) for wl in panels},
      "; median S/N per pixel:", round(float(np.median(1 / sig[(w > 2530) & (w < 2660)])), 0))
