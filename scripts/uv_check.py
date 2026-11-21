"""A test of the ultraviolet line synthesis on A and late-B stars: Fe from unblended Fe II lines in STIS E230H spectra.

The solar test of Section 3.3 covers Fe I lines in the visible. The Tc II thresholds of the A and late-B hosts rest
on the synthesis near 2500 A, where the line haze and the metal bound-free opacity act. For three slowly rotating
stars with STIS E230H spectra and published Fe abundances from the visible (HD 175640, HD 72660, Sirius A), we
synthesize each Fe II line with every line of ions I-III in Kurucz's gfall lists within PAD of it, at solar-scaled
abundances shifted by the star's [Fe/H], and keep the lines that make at least KEEP_FRAC of the absorption within
+/- 1 FWHM of their centre. The synthetic spectrum, broadened by the instrument (R = 114,000) and rotation, is divided by the same
upper envelope as the observed one (scripts/tc_limits.py), and the Fe abundance of each line is the one at which the
synthetic absorption within +/- 1 observed FWHM equals the observed. The mean offset from the published Fe abundance tests
the line strengths that the thresholds use, in the same layers and wavelengths.

    uv run python scripts/uv_check.py     # writes research/uv_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_residuals as MR  # noqa: E402
import tc_detect as D  # noqa: E402
import tc_limits as T  # noqa: E402

from beacon import ROOT  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import C  # noqa: E402
from haze import window_flux  # noqa: E402

CKMS = C / 1e5
HW = 6.0  # A, synthesis window around each Fe II line (the envelope uses a 10 A running window)
PAD = 3.0
STEP = 0.004
KEEP_FRAC = 0.8
WIN = 1.0  # half width of the measured window, in units of the observed line FWHM
VSINI = {"hd175640": 0.5, "hd72660": 6.0, "hd48915": 16.0}  # km/s, paper/data/archival_limits.csv
MIN_DEPTH = 0.15
WMIN, WMAX = 2400.0, 3000.0
OFFSETS = np.arange(-1.0, 1.01, 0.5)
# key, ATLAS9 grid model (Teff, log g), microturbulence (km/s), published Fe (log eps) and its source
STARS = [
    ("hd175640", 12000.0, 4.0, 0.5, 7.20, "Castelli et al. (2015), 7.17-7.23"),
    ("hd72660", 9750.0, 4.0, 2.0, 7.65, "Landstreet et al. (2009)"),
    ("hd48915", 10000.0, 4.5, 2.0, 7.73, "Landstreet et al. (2009)"),
]


def broaden(x, f, fwhm_kms):
    s = x.mean() * fwhm_kms / 2.3548 / CKMS / STEP
    kern = np.exp(-0.5 * (np.arange(-int(5 * s) - 1, int(5 * s) + 2) / s) ** 2)
    return np.convolve(np.pad(f, len(kern) // 2, mode="edge"), kern / kern.sum(), mode="valid")


def main():
    MR.BOOST, MR.HW, MR.PAD = 0.0, HW, PAD
    out = []
    for key, teff, logg, vt, fe_pub, src in STARS:
        d = np.load(T.OUT / key / "stacks.npz")
        w, r, sig = d["e230h_w"], d["e230h_r"], d["e230h_sig"]
        ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
        w, r, sig = w[ok], r[ok], sig[ok]
        b = json.loads((T.OUT / f"{key}.json").read_text())
        fwhm = next(ln["fwhm_kms"] for ln in b["tc_lines"] if ln["instrument"] == "e230h")  # observed line width
        # the synthesis already has the thermal, microturbulent and damping widths: broaden it only by the
        # instrument and rotation
        kern = float(np.hypot(CKMS / 114_000, 1.6 * VSINI[key]))
        atm = L.Atmosphere(teff, logg, vturb=vt, ndepth=72)
        dfe = fe_pub - (atm.s.m.log_abund[26] + 12.0)  # the star's [Fe/H], applied to every metal
        fe2 = L._gfall(26, 1, max(WMIN, w[0] + HW), min(WMAX, w[-1] - HW), min_loggf=-2.0)
        k_fe2 = atm.line_strength(fe2, atm.s.m.log_abund[26] + 12.0)
        # strongest Fe II lines first; neighbours within the window around each
        order = np.argsort(-(fe2.loggf - 5040 / teff * fe2.elow / 8065.54))[:600:4]  # spread over line strength
        cand = np.sort(fe2.wl[order])
        nb, k_sol, mass = MR.select_lines(atm, cand.min() - HW - PAD, cand.max() + HW + PAD, cand)
        rows = []
        for wl0 in cand:
            j = int(np.argmin(np.abs(fe2.wl - wl0)))
            x = np.arange(wl0 - HW, wl0 + HW, STEP)
            nu = C / (x * 1e-8)
            sel = np.abs(nb.wl - wl0) < HW + PAD
            is_fe = sel & (nb.ion == 26)
            is_tgt = is_fe & (np.abs(nb.wl - wl0) < 1e-3)
            k_other = L._profile(atm, wl0, nb.wl[sel & ~is_fe], k_sol[sel & ~is_fe], mass[sel & ~is_fe],
                                 nb.gamma[sel & ~is_fe], nu) * 10**dfe
            k_fe = L._profile(atm, wl0, nb.wl[is_fe & ~is_tgt], k_sol[is_fe & ~is_tgt], mass[is_fe & ~is_tgt],
                              nb.gamma[is_fe & ~is_tgt], nu) * 10**dfe
            k_t = L._profile(atm, wl0, fe2.wl[j:j + 1], k_fe2[j:j + 1], np.array([55.845]), fe2.gamma[j:j + 1], nu)
            kc = atm.kcont(wl0)
            core = np.abs(x - wl0) < WIN * fwhm / CKMS * wl0
            f_all = broaden(x, window_flux(atm, x, kc, [k_other, k_fe, k_t * 10**dfe]), kern)
            f_wo = broaden(x, window_flux(atm, x, kc, [k_other, k_fe]), kern)
            a_all = np.trapezoid(1 - f_all[core], x[core])
            a_t = np.trapezoid(f_wo[core] - f_all[core], x[core])
            if a_t < KEEP_FRAC * a_all or 1 - f_all[core].min() < MIN_DEPTH:
                continue
            pix = D.window(w, wl0 - WIN * fwhm / CKMS * wl0, wl0 + WIN * fwhm / CKMS * wl0)
            big = D.window(w, wl0 - HW, wl0 + HW)
            if len(w[pix]) < 3 or len(w[big]) < 50:
                continue
            obs = np.trapezoid(1 - r[pix], w[pix])
            syn = []
            for off in OFFSETS:  # Fe changes all Fe lines; other metals stay at the star's [Fe/H]
                f = broaden(x, window_flux(atm, x, kc, [k_other, (k_fe + k_t) * 10 ** (dfe + off)]), kern)
                fn = f / T.envelope(x, f)
                syn.append(np.trapezoid(1 - np.interp(w[pix], x, fn), w[pix]))
            syn = np.array(syn)
            if not (syn[0] < obs < syn[-1]):
                continue
            off = float(np.interp(obs, syn, OFFSETS))
            rows.append(dict(wl=float(wl0), loggf=float(fe2.loggf[j]), elow_eV=float(fe2.elow[j] / 8065.54),
                             frac_own=float(a_t / a_all), depth=float(1 - f_all[core].min()), dlog_fe=off))
        offs = np.array([x["dlog_fe"] for x in rows])
        rec = dict(star=key, teff_model=teff, logg_model=logg, vturb=vt, fe_published=fe_pub, source=src,
                   n_lines=len(rows), mean_offset=float(offs.mean()) if len(offs) else None,
                   median_offset=float(np.median(offs)) if len(offs) else None,
                   std=float(offs.std()) if len(offs) else None,
                   sem=float(offs.std() / np.sqrt(len(offs))) if len(offs) > 1 else None, lines=rows)
        out.append(rec)
        print(f"{key}: {len(rows)} unblended Fe II lines; Fe(UV) - Fe(published) = mean {rec['mean_offset']:+.2f}, "
              f"median {rec['median_offset']:+.2f}, std {rec['std']:.2f} dex")
    (ROOT / "research" / "uv_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
