"""A Tc beacon held in the magnetosphere of a magnetic Ap/Bp star.

Radiation pushes Tc outward much harder than gravity pulls it in (research/levitation.json). Along a
closed dipole field line the radial net force has no component at the line's apex, which lies in the
magnetic equatorial plane, so Tc ions collect there (scripts/ap_held.py). Since radiation and gravity both
fall as 1/r^2, the ratio stays the same in the magnetosphere. Gas this thin cannot drag the ions across the
field, and a trace dopant cannot distort the field. The beacon is a thin disk of Tc II in the magnetic
equatorial plane, from the stellar surface to r_out = k R.

Absorption. Seen at angle beta to the disk plane, the front half of the disk covers the part of the stellar
disc with y' > 0 and R^2 <= x^2 + (y' / sin beta)^2 <= k^2 R^2 (x, y' on the sky in units of R, y' across
the projected disk). The column through the disk is Sigma / sin beta. Absorption and detection follow
beacon/comet.py (the comet gas model) with covering factor alpha(beta, k), a velocity spread DV (rigid
corotation of a slowly rotating Ap star across the disc) and an excitation temperature T_EXC.

Cost. The inventory is Sigma m pi R^2 (k^2 - 1). For each tilt beta we choose k to minimize it. The upkeep
replaces decay only, so the result holds only if the disk keeps its Tc singly ionized.

Ionization. Tc II is photoionized by photons above 15.26 eV (below 812 A). Its cross-section is not
known; we give the rate per 1e-18 cm^2 from the blanketed surface flux of Castelli & Kurucz, diluted at
r = 1.5 R. Recombination keeps Tc singly ionized where n_e > rate / ALPHA_REC (n_e_crit). If it does not,
the disk turns to Tc III, whose resonance lines near 1135 A (Kurucz's computed list) a 6 m far-ultraviolet
telescope reads; the Tc III case uses ground-level populations (T_exc = 3000 K, U = 6).

    uv run python scripts/disk_beacon.py     # writes research/disk_beacon.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comet import radius  # noqa: E402

from beacon import ROOT, comet, detect, hosts  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import AMU, RSUN, YR, C, H  # noqa: E402

HOSTS = ((8000, 1.8), (9000, 2.3), (12250, 3.4))
INST = "2.4 m space UV, R = 114,000"
SNR = 300.0
DV = 20.0  # km/s: 2 v_eq for a 2 Rsun star rotating in 5 d
T_EXC = (None, 3000.0)  # None: the star's Teff (radiative regime, as for comets); 3000 K: mostly ground levels
BETAS = (2.0, 5.0, 10.0, 20.0, 30.0, 45.0)
KS = np.geomspace(1.05, 30.0, 60)
SIGMA_PI = 1e-18  # cm^2, assumed scale of the Tc II photoionization cross-section
R_ION = 1.5
IP_TC2 = 15.26  # eV
ALPHA_REC = 3e-12  # cm^3 s^-1, assumed recombination coefficient of Tc III near 1e4 K
FUV = "6 m space UV-optical, R = 100,000"


def tc3_species():
    import csv

    from beacon import KURUCZ, atoms
    with (KURUCZ / "tc" / "tc2_lines.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["loggf"]) >= -1.0]
    lines = [atoms.Line(float(r["wl_A"]), float(r["loggf"]), float(r["elow_cm"]), 43, 2) for r in rows]
    return comet.Species("Tc III", 43, lines, sp_mass(), ion=2)


def sp_mass():
    from beacon import atoms
    return atoms.MASS[43]

_g = np.linspace(-1, 1, 801)
_X, _Y = np.meshgrid(_g, _g)
_ON = _X**2 + _Y**2 <= 1


def alpha(beta_deg, k):
    s = np.sin(np.radians(beta_deg))
    q = _X**2 + (_Y / s) ** 2
    cov = _ON & (_Y > 0) & (q >= 1) & (q <= k * k)
    return cov.sum() / _ON.sum()


def photoionization_rate(teff, logg):
    """Tc II photoionizations per second per ion at r = R_ION R, for a cross-section SIGMA_PI above threshold."""
    w, fnu, _ = L.castelli_flux(teff, logg)  # A, F_nu (erg cm^-2 s^-1 Hz^-1) at the surface
    lam_max = 12398.4 / IP_TC2
    sel = w < lam_max
    nu = C / (w[sel] * 1e-8)
    dil = 0.5 * (1 - np.sqrt(1 - 1 / R_ION**2))
    # 4 pi J = 4 W F for a uniformly bright disc (J = W I, I = F / pi)
    n_phot = np.trapezoid((4 * dil * fnu[sel] / (H * nu))[::-1], nu[::-1])
    return float(n_phot * SIGMA_PI)


def design(sp, inst, R, t_exc):
    """Smallest inventory (g) for each tilt beta, with the disk size k chosen to minimize it."""
    rows = {}
    for b in BETAS:
        s = np.sin(np.radians(b))
        best = None
        for k in KS:
            a = alpha(b, k)
            if a <= 0:
                continue
            n = comet.threshold(sp, inst, SNR, alpha=a, dv_kms=DV, t_exc=t_exc)  # column along the sightline
            m = n * s * sp.mass * AMU * np.pi * R**2 * (k * k - 1)
            if best is None or m < best[0]:
                best = (m, k, a, n)
        m, k, a, n = best
        p = hosts.upkeep_power(m * hosts.NUCLIDES["99Tc"]["A"] / sp.mass, "99Tc capture") / 1e7
        rows[f"{b:g}"] = dict(k=float(k), alpha=float(a), column=float(n), mass_kg=m / 1e3, upkeep_W=p,
                              sagan_K=(np.log10(p) - 6) / 10)
    return rows


def main():
    sp = comet.species()["Tc II"]
    sp3 = tc3_species()
    insts = {i.name: i for i in detect.INSTRUMENTS}
    inst = insts[INST]
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    out = dict(inst=INST, snr=SNR, dv_kms=DV, sigma_pi=SIGMA_PI, hosts={})
    for teff, mass in HOSTS:
        r_rsun = radius(teff)
        R = r_rsun * RSUN
        gam = photoionization_rate(teff, 4.0)
        rec = dict(mass_msun=mass, r_rsun=r_rsun, g_rad_over_g=10 ** min(lev[f"tc|{teff}|+0"]["log_grad_diffw"]) / 1e4,
                   photoionization_rate_s=gam, t_ion_yr=1 / gam / YR if gam > 0 else None, cases={})
        for t_exc in T_EXC:
            te = float(teff) if t_exc is None else t_exc
            rec["cases"]["Tc II, T_exc = " + ("Teff" if t_exc is None else f"{t_exc:g} K")] = design(sp, inst, R, te)
        rec["n_e_crit"] = gam / ALPHA_REC
        rec["cases"]["Tc III, 6 m FUV"] = design(sp3, insts[FUV], R, 3000.0)
        out["hosts"][str(teff)] = rec
        print(f"{teff} K, R = {r_rsun:.2f} Rsun, g_rad/g >= {rec['g_rad_over_g']:.0f}, "
              f"Tc II photoionization time {rec['t_ion_yr']:.2g} yr for sigma = {SIGMA_PI:g} cm^2, n_e,crit {rec['n_e_crit']:.1g}")
        for case, rows in rec["cases"].items():
            print(f"  {case}: " + "  ".join(f"beta {b}: {x['mass_kg']:.1e} kg K={x['sagan_K']:.2f} (k {x['k']:.1f}, "
                                                    f"cover {x['alpha']:.2f})" for b, x in rows.items()))
    (ROOT / "research" / "disk_beacon.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
