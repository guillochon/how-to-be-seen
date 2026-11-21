"""How well a model of HD 175640 predicts its spectrum at the Tc lines, and the Tc a model-based observer detects.

The detection test of scripts/tc_detect.py uses only Tc lines on clean continuum. An observer with a model of the
star can use every line, to the accuracy of the model. This script builds the model a modeller would build
today and measures that accuracy in the HST STIS E230H spectrum.

1. Model. LTE synthesis in the ATLAS9 model nearest the star (12,000 K, log g = 4), with the lines of ions I-III
   of every element from He to U in Kurucz's gfall lists (scripts/fetch_gfall.py) whose centre opacity would
   exceed RATIO_MIN of the continuum somewhere above tau5000 = 3 at 1e6 times the solar abundance. Ionization
   and partition functions are Kurucz's, or NIST ground levels where Kurucz has none (beacon/levitation.py).
   Each element has one free abundance offset from the model's solar-scaled value. The synthesis is broadened by
   a Gaussian of free width, shifted by a free velocity, and multiplied by a straight-line continuum in each
   window.
2. Fit. Windows of +/- HW Angstrom around each Tc line in the stack. The abundances and continua are fitted by
   least squares to the stack, leaving out +/- MASK_KMS around every Tc line.
3. Residuals. q = r / model. The RMS of q - 1 against the photon noise of the stack measures the accuracy.
4. Detection. As in scripts/tc_detect.py, with the model residual 1 - q in place of 1 - r / continuum and every
   Tc line in place of the clean lines: the template-weighted depth at each line, its noise from N_NULL random
   positions in the same window away from the Tc lines, and the abundance at which the sum rises by z = 5
   above its value without Tc.

    uv run python scripts/model_residuals.py     # writes research/model_residuals.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_detect as D  # noqa: E402
import tc_limits as T  # noqa: E402

from beacon import ROOT, atoms  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import AMU, C, K  # noqa: E402
from beacon.synth import _flux, planck_lambda  # noqa: E402

KEY, INST = "hd175640", "e230h"
TEFF, LOGG = 12000.0, 4.0
VTURB = 0.5  # km/s: a quiet atmosphere
RATIO_MIN = 1e-3
BOOST = 6.0  # dex above solar at which lines are selected
HW = 3.0  # A, half width of each window
PAD = 1.5  # A, lines this far outside a window still add their wings
STEP = 0.004  # A, synthesis grid
MASK_KMS = 15.0
N_NULL = 400
CKMS = C / 1e5
RNG = np.random.default_rng(1)


def select_lines(atm, wmin, wmax, centres):
    """Lines of every species near the windows (Z in the ion field), their integrated opacity [n, ndepth] at the
    model's solar-scaled abundance, and their atomic masses."""
    with (L.KURUCZ / "nist_weights.csv").open() as fh:
        weight = {int(r["z"]): float(r["weight"]) for r in csv.DictReader(fh)}
    deep = atm.tau5 < 3.0
    keep = []
    for path in sorted((L.KURUCZ / "gfall").glob("gf????.all")):
        z, ion = int(path.stem[2:4]), int(path.stem[4:6])
        if z in (1, 43) or ion > 2:
            continue
        ls = L._gfall(z, ion, wmin, wmax, min_loggf=-5.0)
        near = np.min(np.abs(ls.wl[:, None] - centres[None, :]), axis=1) < HW + PAD if len(ls.wl) else []
        if not np.any(near):
            continue
        ls = L.LineSet(z, ls.ion[near], ls.wl[near], ls.loggf[near], ls.elow[near], ls.gamma[near])
        k = atm.line_strength(ls, atm.s.m.log_abund[z] + 12.0)
        dnu = 1e8 / ls.wl[:, None] * np.sqrt(2 * K * atm.T / (weight[z] * AMU) + atm.vturb**2)[None, :]
        ratio = (10**BOOST * k / (np.sqrt(np.pi) * dnu) / atm.kcont(ls.wl.mean())[None, :])[:, deep].max(axis=1)
        sel = ratio > RATIO_MIN
        if sel.any():
            keep.append((np.full(sel.sum(), z), ls.wl[sel], k[sel], np.full(sel.sum(), weight[z]), ls.gamma[sel]))
    zz, wl, k, m, g = (np.concatenate(x) for x in zip(*keep, strict=True))
    o = np.argsort(wl)
    return L.LineSet(0, zz[o], wl[o], np.zeros(len(wl)), np.zeros(len(wl)), g[o]), k[o], m[o]


def main():
    star = T.STARS[KEY]
    b = json.loads((T.OUT / f"{KEY}.json").read_text())
    d = np.load(T.OUT / KEY / "stacks.npz")
    w, r, sig = d[INST + "_w"], d[INST + "_r"], d[INST + "_sig"]
    ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
    w, r, sig = w[ok], r[ok], sig[ok]
    recs = [ln for ln in b["tc_lines"] if ln["instrument"] == INST]
    fwhm = recs[0]["fwhm_kms"]
    lines = [atoms.Line(ln["wl"], ln["loggf"], ln["elow"], 43, 0 if ln["species"] == "Tc I" else 1) for ln in recs]
    lines = [ln for ln in lines if w[0] + HW < ln.wl < w[-1] - HW]
    tc_wl = np.array([ln.wl for ln in lines])
    syn = T.synths(star)[1.0]
    prof = {ln.wl: T.Profiles(syn, ln, fwhm) for ln in lines}

    atm = L.Atmosphere(TEFF, LOGG, vturb=VTURB, ndepth=72)
    nb, k_sol, mass = select_lines(atm, tc_wl.min() - HW - PAD, tc_wl.max() + HW + PAD, tc_wl)
    elements = sorted(set(int(z) for z in nb.ion))  # the LineSet stores Z in the ion field

    # per window: wavelength grid, continuum opacity, and each element's line opacity at solar abundance
    wins = []
    for ln in lines:
        x = np.arange(ln.wl - HW - 0.3, ln.wl + HW + 0.3, STEP)
        nu = C / (x * 1e-8)
        sel = np.abs(nb.wl - ln.wl) < HW + PAD
        kel = {}
        for z in elements:
            m = sel & (nb.ion == z)
            if m.any():
                kel[z] = L._profile(atm, ln.wl, nb.wl[m], k_sol[m], mass[m], nb.gamma[m], nu) / atm.rho[:, None]
        kc = (atm.kcont(ln.wl) / atm.rho)[:, None]
        S = planck_lambda(ln.wl, atm.T)[:, None] * np.ones((1, len(x)))
        tauc = np.concatenate([[kc[0, 0] * atm.rhox[0]], kc[0, 0] * atm.rhox[0]
                               + np.cumsum(0.5 * (kc[1:, 0] + kc[:-1, 0]) * np.diff(atm.rhox))])
        Fc = _flux(tauc[:, None], planck_lambda(ln.wl, atm.T)[:, None])[0]
        pix = D.window(w, ln.wl - HW, ln.wl + HW)
        fit = np.abs((w[pix] / ln.wl - 1) * CKMS)[:, None] > MASK_KMS
        fit = fit[:, 0] & np.all(np.abs((w[pix][:, None] / tc_wl[None, :] - 1) * CKMS) > MASK_KMS, axis=1)
        wins.append(dict(ln=ln, x=x, kel=kel, kc=kc, S=S, Fc=Fc, pix=pix, fit=fit))
    print(f"{len(nb.wl)} neighbour lines, elements {elements}; {len(wins)} windows")

    def synth(win, dlog, fw, dv):
        k = win["kc"] + sum(10.0 ** dlog[z] * kk for z, kk in win["kel"].items())
        tau0 = k[:1] * atm.rhox[0]
        tau = np.vstack([tau0, tau0 + np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(atm.rhox)[:, None], axis=0)])
        f = _flux(tau, win["S"]) / win["Fc"]
        x = win["x"]
        s = x.mean() * fw / 2.3548 / CKMS / STEP
        kern = np.exp(-0.5 * (np.arange(-int(5 * s) - 1, int(5 * s) + 2) / s) ** 2)
        f = np.convolve(np.pad(f, len(kern) // 2, mode="edge"), kern / kern.sum(), mode="valid")
        return np.interp(w[win["pix"]] / (1 + dv / CKMS), x, f)

    def cont_fit(win, mod, data):
        """Straight-line continuum c0 + c1 (w - wl) that best scales mod to data on the fitted pixels."""
        xx = w[win["pix"]] - win["ln"].wl
        f = win["fit"]
        A = np.vstack([mod[f], mod[f] * xx[f]]).T
        c = np.linalg.lstsq(A / sig[win["pix"]][f, None], data[f] / sig[win["pix"]][f], rcond=None)[0]
        return mod * (c[0] + c[1] * xx)

    def unpack(p):
        return dict(zip(elements, p[:-2], strict=True)), p[-2], p[-1]

    def resid(p):
        dlog, fw, dv = unpack(p)
        out = []
        for win in wins:
            data = r[win["pix"]]
            mod = cont_fit(win, synth(win, dlog, fw, dv), data)
            out.append(((data - mod) / sig[win["pix"]])[win["fit"]])
        return np.concatenate(out)

    n = len(elements)
    p0 = np.concatenate([np.zeros(n), [fwhm * 0.8, 0.0]])
    lo = np.concatenate([np.full(n, -4.0), [1.0, -5.0]])
    hi = np.concatenate([np.full(n, BOOST), [15.0, 5.0]])
    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale=0.5, diff_step=1e-3)
    dlog, fw, dv = unpack(sol.x)
    print(f"broadening FWHM {fw:.2f} km/s, shift {dv:+.2f} km/s; abundance offsets from solar-scaled:",
          {z: round(float(v), 2) for z, v in dlog.items()})

    model = np.ones_like(r)
    for win in wins:
        model[win["pix"]] = cont_fit(win, synth(win, dlog, fw, dv), r[win["pix"]])

    rec = dict(star=star.name, instrument=INST, fwhm_kms=fwhm, fit_fwhm_kms=float(fw), fit_shift_kms=float(dv),
               n_lines=len(nb.wl), dlog={str(z): float(v) for z, v in dlog.items()},
               chi2_per_pixel=float(np.mean(resid(sol.x) ** 2)), lines={})
    for win in wins:
        pix, f = win["pix"], win["fit"]
        q = r[pix] / model[pix]
        rms = float(np.std(q[f] - 1))
        rms_flux = float(np.std((r[pix] - model[pix])[f]))  # in units of the continuum
        noise = float(np.median(sig[pix][f] / model[pix][f]))
        core = np.abs((w[pix] / win["ln"].wl - 1) * CKMS) <= fwhm
        lr = dict(rms=rms, rms_flux=rms_flux, photon=noise, rms_over_photon=rms / noise, core_resid=float(np.mean(q[core] - 1)),
                  clean=D.is_clean(w, r, win["ln"].wl, fwhm))
        rec["lines"][f"{win['ln'].wl:.3f}"] = lr
        print(f"  {win['ln'].wl:9.3f}  clean {str(lr['clean']):5s}  residual RMS {100 * rms:.2f}%  "
              f"flux RMS {100 * rms_flux:.2f}%  photon {100 * noise:.2f}%  ratio {rms / noise:.1f}  mean residual in core {100 * lr['core_resid']:+.2f}%")

    # detection with every Tc line against the model
    hv = fwhm
    stats = {}
    for win in wins:
        ln = win["ln"]
        p = prof[ln.wl]
        sel = np.abs(p.v) <= hv
        tv, td = p.v[sel], 1 - p(D.LE_REF)[sel]
        q = r / np.where(model > 0, model, 1)
        cand = RNG.uniform(ln.wl - HW + 0.3, ln.wl + HW - 0.3, 6 * N_NULL)
        far = np.min(np.abs(cand[:, None] - tc_wl[None, :]) / tc_wl[None, :] * CKMS, axis=1) > 2 * fwhm
        a_null = np.array([a for a in (D.amplitude(w, q, 1.0, x, tv, td) for x in cand[far][:N_NULL]) if a is not None])
        stats[ln.wl] = (tv, td, float(np.median(a_null)), float(np.std(a_null)))

    def inject(le):
        rr = r.copy()
        for ln in lines:
            p = prof[ln.wl]
            k = D.window(w, ln.wl * (1 - p.v[-1] / CKMS), ln.wl * (1 + p.v[-1] / CKMS))
            rr[k] *= np.interp((w[k] / ln.wl - 1) * CKMS, p.v, p(le))
        return rr

    def z_model(le, rr=None):
        rr = inject(le) if rr is None else rr
        q = rr / np.where(model > 0, model, 1)
        num = den = 0.0
        for ln in lines:
            tv, td, mu, s = stats[ln.wl]
            a = D.amplitude(w, q, 1.0, ln.wl, tv, td)
            p = prof[ln.wl]
            A = float(np.sum(td * (1 - np.interp(tv, p.v, p(le)))) / np.sum(td * td))
            num += A / s**2 * (a - mu)
            den += (A / s**2) ** 2 * s**2
        return num / np.sqrt(den)

    z0 = z_model(D.LE_REF, r)
    grid = np.arange(-3.0, 8.01, 0.5)
    hit = next(g for g in grid if z_model(g) - z0 >= D.Z)
    a_, b_ = hit - 0.5, hit
    for _ in range(8):
        m = 0.5 * (a_ + b_)
        a_, b_ = (m, b_) if z_model(m) - z0 < D.Z else (a_, m)
    rec.update(z0=float(z0), le_model=round(0.5 * (a_ + b_), 2))
    row = next(x for x in csv.DictReader((T.OUT / "detect.csv").open()) if x["key"] == KEY and x["instrument"] == INST)
    rec.update(le_photon=float(row["le_photon"]), le_clean=float(row["le_clean"]), limit=float(row["limit"]))
    print(f"model-based detection: log eps {rec['le_model']} (z0 {z0:+.1f}); photon ideal {rec['le_photon']}, "
          f"clean lines only {rec['le_clean']}, archival limit {rec['limit']}")
    (ROOT / "research" / "model_residuals.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
