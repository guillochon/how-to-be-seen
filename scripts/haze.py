"""How much the lines of other elements weaken the Tc lines at the detection threshold.

The thresholds of Section 4 synthesize each Tc line alone. In a real star the Tc line forms among the lines of other
elements, which raise the opacity around it and lower the flux that it can absorb; in the ultraviolet of A stars
this line haze removes 20 to 45 per cent of the continuum flux. Here each Tc line is synthesized among every line of
ions I-III in Kurucz's gfall lists (scripts/model_residuals.py selection, at the model's solar-scaled abundances)
within PAD of it, with the metal bound-free opacity of beacon/metals.py. The signal that a receiver who knows the
star's spectrum measures is the change in equivalent width that the Tc makes,
    dW = int (F_without Tc - F_with Tc) / F_cont dlambda,
which convolution with the instrument does not change. Its ratio to the equivalent width of the isolated line, at
the threshold abundance of research/results.json, scales each line's significance in Equation (5), and the
threshold moves by -log10(Z_haze / Z_alone) in the weak-line limit.

    uv run python scripts/haze.py      # writes research/haze.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_residuals as MR  # noqa: E402

from beacon import ROOT, atoms, detect, hosts  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import C  # noqa: E402
from beacon.synth import _flux, planck_lambda  # noqa: E402

HW = 1.0  # A, half width of the window in which dW is integrated
PAD = 12.0  # A, lines this far from the Tc line add their wings
STEP = 0.003  # A
CASES = [  # host, species, instrument
    ("G", "Tc I", "8.2 m, R = 140,000"),
    ("F", "Tc I", "8.2 m, R = 140,000"),
    ("Am", "Tc II", "2.4 m space UV, R = 114,000"),
    ("Am", "Tc II", "8.2 m UV-blue, R = 80,000"),
    ("A0", "Tc II", "2.4 m space UV, R = 114,000"),
    ("B", "Tc II", "2.4 m space UV, R = 114,000"),
    ("B", "Tc II", "8.2 m UV-blue, R = 80,000"),
]


def window_flux(atm, x, kc, k_list):
    """Normalized flux on wavelengths x with continuum kc [nd] and line opacities k_list ([nd, nx] each, per volume)."""
    k = (kc[:, None] + sum(k_list)) / atm.rho[:, None]
    tau0 = k[:1] * atm.rhox[0]
    tau = np.vstack([tau0, tau0 + np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(atm.rhox)[:, None], axis=0)])
    S = planck_lambda(x.mean(), atm.T)[:, None] * np.ones((1, len(x)))
    kcr = kc / atm.rho
    tauc = np.concatenate([[kcr[0] * atm.rhox[0]], kcr[0] * atm.rhox[0] + np.cumsum(0.5 * (kcr[1:] + kcr[:-1]) * np.diff(atm.rhox))])
    Fc = _flux(tauc[:, None], planck_lambda(x.mean(), atm.T)[:, None])[0]
    return _flux(tau, S) / Fc


def main():
    res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    curves = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))
    H = {h.key: h for h in hosts.host_grid()}
    INST = {i.name: i for i in detect.INSTRUMENTS}
    MR.BOOST, MR.HW, MR.PAD = 0.0, HW, PAD
    out = []
    for hk, sp, iname in CASES:
        h, inst, c = H[hk], INST[iname], curves[hk][sp]
        le = res[hk]["species"][sp][f"{iname}|300"]["log_eps"]
        model = "sun" if h.solar else None
        atm = L.Atmosphere(h.teff, h.logg, vturb=h.vturb, ndepth=72) if not model else None
        if model:  # the Sun uses Kurucz's solar model, as the thresholds do
            atm = L.Atmosphere.__new__(L.Atmosphere)
            from beacon import atlas
            from beacon.synth import Synth
            atm.s = Synth(atlas.sun(), vturb=h.vturb, nrefine=72)
            atm.T, atm.ne, atm.rho, atm.rhox = atm.s.m.T, atm.s.m.ne, atm.s.st.rho, atm.s.m.rhox
            atm.vturb, atm.nH, atm.logg = h.vturb * 1e5, atm.s.st.nH, h.logg
            atm.tau5 = atm.s.tau_cont_500()
            atm._frac, atm._kc = {}, {}
        lines = [(i, ln) for i, ln in enumerate(c.lines) if inst.wmin <= ln.wl <= inst.wmax]
        centres = np.array([ln.wl for _, ln in lines])
        nb, k_sol, mass = MR.select_lines(atm, centres.min() - HW - PAD, centres.max() + HW + PAD, centres)
        z2_alone = z2_haze = 0.0
        rows = []
        for i, ln in lines:
            x = np.arange(ln.wl - HW, ln.wl + HW, STEP)
            nu = C / (x * 1e-8)
            sel = np.abs(nb.wl - ln.wl) < HW + PAD
            k_nb = L._profile(atm, ln.wl, nb.wl[sel], k_sol[sel], mass[sel], nb.gamma[sel], nu)
            tc = L.LineSet(43, np.array([ln.ion]), np.array([ln.wl]), np.array([ln.loggf]), np.array([ln.elow]),
                           np.array([[ln.gamma_rad or 0.0, ln.gamma_stark or 0.0, ln.gamma_waals or 0.0]]))
            k_tc = L._profile(atm, ln.wl, tc.wl, atm.line_strength(tc, le), np.array([atoms.MASS[43]]), tc.gamma, nu)
            kc = atm.kcont(ln.wl)
            f_alone = window_flux(atm, x, kc, [k_tc])
            f0 = window_flux(atm, x, kc, [k_nb])
            f1 = window_flux(atm, x, kc, [k_nb, k_tc])
            w_alone = 1e3 * np.trapezoid(1 - f_alone, x)
            dw = 1e3 * np.trapezoid(f0 - f1, x)
            sig = detect.sigma_ew(ln.wl, inst.R, 300.0, detect.observed_fwhm(ln.wl, inst.R, c.fwhm_int[i], h.vsini))
            z2_alone += (w_alone / sig) ** 2
            z2_haze += (dw / sig) ** 2
            rows.append(dict(wl=ln.wl, w_alone_mA=w_alone, dw_mA=dw, ratio=dw / w_alone if w_alone > 0 else np.nan,
                             f_center_without_tc=float(np.interp(ln.wl, x, f0)), z_alone=w_alone / sig))
        shift = -np.log10(np.sqrt(z2_haze / z2_alone))
        z_weights = np.array([r["z_alone"] ** 2 for r in rows])
        ratios = np.array([r["ratio"] for r in rows])
        rec = dict(host=hk, species=sp, instrument=iname, log_eps=le, n_lines=len(rows), n_neighbours=len(nb.wl),
                   weighted_ratio=float(np.sum(z_weights * ratios) / np.sum(z_weights)),
                   threshold_shift_dex=float(shift), lines=rows)
        out.append(rec)
        print(f"{hk:3s} {sp:6s} {iname:30s} log eps {le:+.2f}: {len(rows)} lines, {len(nb.wl)} neighbours; "
              f"Z-weighted dW/W = {rec['weighted_ratio']:.2f}; threshold shift {shift:+.2f} dex")
        for r in sorted(rows, key=lambda r: -r["z_alone"])[:4]:
            print(f"      {r['wl']:.2f}: W alone {r['w_alone_mA']:.2f} mA, dW {r['dw_mA']:.2f} mA, "
                  f"ratio {r['ratio']:.2f}, flux at centre without Tc {r['f_center_without_tc']:.2f}")
    (ROOT / "research" / "haze.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
