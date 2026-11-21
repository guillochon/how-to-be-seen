"""How far a Tc beacon in a quiet late-B star can be seen on a given power budget, what fraction of the
Galaxy's star systems lie within that reach once dust is included, and what a laser network costs to
reach the same systems.

The beacon spends a power P on breeding 99Tc by neutron capture; the inventory it holds against
decay (research/quiet_host.json, no wind) sets the abundance. The lowest S/N per pixel at which that
abundance reaches Z = 5 sets, through a 10 h photon-limited exposure, the dust-free distance d0 to
which the beacon is seen. A star at distance d behind an extinction A_lambda is reached if
d 10^(0.2 A_lambda) <= d0.

Stars: the Galaxy model of scripts/broadcast.py (Bland-Hawthorn & Gerhard 2016), host at the Sun.
Dust: the axisymmetric disk of Drimmel & Spergel (2001; rho0 = 1098 MJy/sr/kpc, h_r = 2.26 kpc,
h_d = 134.4 pc flaring by 14.8 pc/kpc beyond 4.4 kpc, central hole, kappa_V = 0.0180 (MJy/sr)^-1),
scaled by 0.5, 1 and 2; extinction law of Gordon et al. (2023) with R_V = 3.1.
Systems per solar mass: 339 systems within 10 pc (Reyle et al. 2021) over the local stellar density
0.043 Msun/pc^3 (McKee et al. 2015).
Laser: 1 MW at 1315 nm through a 39 m aperture per target system, which Clark & Cahoy (2018) find
detectable from a Sun-like host by a 1 m telescope to about 20,000 lt-yr. Their criterion is that the
laser adds 0.1 per cent to its star's light in a 1 nm band, so a laser on the late-B host needs that
power times the host's luminosity at 1315 nm over the Sun's (Castelli & Kurucz 2003 fluxes; the Sun
as the 5750 K, log g 4.5 model).

    uv run python scripts/reach_budget.py     # writes research/reach_budget.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from dust_extinction.parameter_averages import G23
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import broadcast  # noqa: E402

from beacon import ROOT, detect, hosts, survey  # noqa: E402
from beacon.levitation import castelli_flux  # noqa: E402

P_HUMANITY = 1.7e13  # W, primary power in 2015 (Gray 2020)
T_EXP = 10 * 3600.0
INSTRUMENTS = ("2.4 m space UV, R = 114,000", "6 m space UV-optical, R = 100,000")
DUST_SCALES = (0.5, 1.0, 2.0)
N_STARS = 200_000
N_STEPS = 400
SYSTEMS_PER_MSUN = 339 / (4 / 3 * np.pi * 10.0**3) / 0.043
LASER_W, LASER_WL = 1e6, 13150.0  # W per target from a Sun-like host, A
RSUN_CM, G_CGS = 6.957e10, 6.674e-8
REACH_KPC = 5.0
RHO0, HR, H0, H1, RF, KAPPA_V, RSUN_DS = 1098.0, 2.26, 0.1344, 0.0148, 4.4, 0.0180, 8.0


def dust(r, z):
    """Drimmel & Spergel (2001) axisymmetric dust density (MJy/sr/kpc) at Galactocentric radius r, height z (kpc)."""
    h = np.where(r > RF, H0 + H1 * (r - RF), H0)
    rho = RHO0 * np.exp(-r / HR) / np.cosh(z / h) ** 2
    hole = RHO0 * np.exp(-0.5 * RSUN_DS / HR) * np.exp(-((r / RSUN_DS - 0.5) ** 2) / 0.25**2) / np.cosh(z / H0) ** 2
    return np.where(r < 0.5 * RSUN_DS, hole, rho)


def extinction_v(x, y, z):
    """A_V (mag) from a host at the Sun's position to stars at x, y, z (kpc)."""
    x0, z0 = broadcast.R0_KPC, broadcast.Z0_KPC
    d = np.sqrt((x - x0) ** 2 + y**2 + (z - z0) ** 2)
    t = (np.arange(N_STEPS) + 0.5) / N_STEPS
    out = np.empty_like(d)
    for i in range(0, len(d), 5000):
        s = slice(i, i + 5000)
        px = x0 + np.outer(x[s] - x0, t)
        py = np.outer(y[s], t)
        pz = z0 + np.outer(z[s] - z0, t)
        out[s] = 1.086 * KAPPA_V * dust(np.hypot(px, py), pz).mean(axis=1) * d[s]
    return d, out


def reach(curve, inst, mv: float, log_eps: float) -> tuple[float, float, float]:
    """(S/N per pixel needed, wavelength of the median covered line in A, dust-free reach in pc)."""
    f = lambda ls: survey.required_log_eps(curve, inst, 10**ls) - log_eps  # noqa: E731
    snr = 10 ** brentq(f, np.log10(2.0), np.log10(1000.0), xtol=1e-3)
    sel = [(ln, fr) for ln, fr in zip(curve.lines, curve.flux_ratio, strict=True) if inst.wmin <= ln.wl <= inst.wmax]
    ln, fr = sorted(sel, key=lambda t: t[0].wl)[len(sel) // 2]
    g = lambda logd: np.log10(detect.snr_per_pixel(inst, mv + 5 * (logd - 1), T_EXP, ln.wl, fr) / snr)  # noqa: E731
    return snr, ln.wl, 10 ** brentq(g, -3, 6)


def host_ratio(teff: float, logg: float, mstar: float) -> float:
    """Luminosity per unit wavelength at LASER_WL of a host relative to the Sun."""
    def lum(t, g, r_cm):
        w, f, _ = castelli_flux(t, g)
        return np.interp(LASER_WL, w, f) * r_cm**2
    r = np.sqrt(G_CGS * mstar * hosts.MSUN / 10**logg)
    return float(lum(teff, logg, r) / lum(5750, 4.5, RSUN_CM))


def main():
    q = json.loads((ROOT / "research" / "quiet_host.json").read_text())
    host = hosts.Host("B", "quiet late-B", q["teff"], q["logg"], q["mstar"], q["mmix_msun"], q["mmix_msun"],
                      q["mmix_msun"], 1.0, 10.0, 0.0)
    curve = survey.build_curves(host)["Tc II"]
    mv = survey.absolute_v(host.teff)
    x, y, z = broadcast.stars(np.random.default_rng(1), N_STARS)
    d, av = extinction_v(x, y, z)
    law = G23(Rv=3.1)
    total = sum(c[1] for c in broadcast.COMPONENTS) * SYSTEMS_PER_MSUN
    per_g = hosts.upkeep_power(1.0, "99Tc capture") / 1e7  # W per gram held
    budgets = {"threshold": q["cases"]["no wind"]["upkeep_W"], "humanity": P_HUMANITY}
    out = dict(systems_per_msun=SYSTEMS_PER_MSUN, systems_total=total, av_per_kpc_local=float(
        1.086 * KAPPA_V * dust(np.array([broadcast.R0_KPC]), np.array([0.0]))[0]), rows=[])
    for bname, p in budgets.items():
        m = p / per_g
        le = float(np.log10(m / ((98.906 / hosts.M_H) * hosts.X_HYDROGEN * q["mmix_msun"] * hosts.MSUN)) + 12)
        for name in INSTRUMENTS:
            inst = {i.name: i for i in detect.INSTRUMENTS}[name]
            try:
                snr, wl, d0 = reach(curve, inst, mv, le)
            except ValueError:
                continue
            ratio = float(law(wl * u.AA))
            row = dict(budget=bname, power_W=p, sagan_K=(np.log10(p) - 6) / 10, mass_kg=m / 1e3, log_eps=le,
                       instrument=name, snr=snr, wl_A=wl, a_over_av=ratio, d0_pc=d0, fraction={})
            for k in DUST_SCALES:
                ok = d * 10 ** (0.2 * k * av * ratio) <= d0 / 1e3
                row["fraction"][f"{k:g}"] = float(ok.mean())
            row["fraction_no_dust"] = float((d <= d0 / 1e3).mean())
            out["rows"].append(row)
            print(f"{bname:9s} {name:34s} P = {p:.1e} W  S/N = {snr:6.1f}  {wl:.0f} A  A/AV = {ratio:.2f}  "
                  f"d0 = {d0 / 1e3:6.2f} kpc  no dust {row['fraction_no_dust']:.3f}  dust x0.5/1/2 " +
                  " ".join(f"{row['fraction'][f'{k:g}']:.3f}" for k in DUST_SCALES), flush=True)
    n5 = float((d <= REACH_KPC).mean()) * total
    hr = host_ratio(q["teff"], q["logg"], q["mstar"])
    out["laser"] = dict(reach_kpc=REACH_KPC, systems=n5, host_ratio=hr, rows={})
    for name, per in (("Sun-like host", LASER_W), ("late-B host", LASER_W * hr)):
        tot = n5 * per
        out["laser"]["rows"][name] = dict(power_per_target_W=per, power_W=tot, sagan_K=(np.log10(tot) - 6) / 10,
                                          over_humanity=tot / P_HUMANITY, duty_at_humanity=P_HUMANITY / tot)
        print(f"laser from {name}: {n5:.1e} systems within {REACH_KPC} kpc x {per:.1e} W = {tot:.1e} W, "
              f"K = {(np.log10(tot) - 6) / 10:.2f}, {tot / P_HUMANITY:.0f} x humanity", flush=True)
    print(f"local A_V = {out['av_per_kpc_local']:.2f} mag/kpc; {total:.1e} systems in the Galaxy; host ratio at "
          f"{LASER_WL / 1e4:.3f} um = {hr:.1f}")
    (ROOT / "research" / "reach_budget.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
