"""A gas beacon aimed at many stars: a belt of neutral Tc around a Sun-like star in the Galactic plane.

Writes research/broadcast.json and paper/tables/broadcast.tex.

Geometry. A shell zone of half-width Theta about the Galactic plane, released at rest at radius r0
from the host, is seen in front of the stellar disc by every observer within Galactic latitude
|b| < Theta (for Theta much larger than R*/r0). Each observer looks through the zone where it is
thin, so the column is the surface density N0 of the zone, and the whole disc is covered
(alpha = 1). The mass is N0 m 4 pi r0^2 sin(Theta); Theta = 90 degrees is an isotropic shell.

Evolution. Each part of the zone moves radially outward under radiation pressure (beta > 1), so
the zone keeps its angular extent; spreading within the zone is replenished by its neighbours.
The column falls as (r0 / r)^2 and photoionization removes neutral Tc with lifetime
tau_ion(r) = tau_ion(1 AU) (r / AU)^2 (Mn analog, scripts/comet_opt.py). With N0 = X N_thr the
beacon is visible until X (r0/r)^2 S(t) = 1, where S is the surviving fraction. We choose X and
r0 to minimize the cost, X / t_vis, with the same monitoring metric as scripts/comet_opt.py.

Coverage. The fraction of the Galaxy's stars within |b| < Theta, seen from a host at R0 = 8.2 kpc
and z0 = 25 pc, for a Monte Carlo model of the thin disk, thick disk and bulge with the parameters
of Bland-Hawthorn & Gerhard (2016): thin disk 3.5e10 Msun, R = 2.6 kpc, z = 300 pc; thick disk
6e9 Msun, R = 2.0 kpc, z = 900 pc; bulge 1.5e10 Msun with vertical scale height 180 pc and an
axisymmetric exponential scale length of 0.55 kpc (the geometric mean of the bar's 0.70 and
0.44 kpc). Stars are counted in proportion to stellar mass.

    uv run python scripts/broadcast.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comet_opt as co  # noqa: E402

from beacon import ROOT, comet, detect, hosts, survey  # noqa: E402
from beacon.const import AMU, AU, MSUN, RSUN, YR, C  # noqa: E402

R0_KPC, Z0_KPC = 8.2, 0.025
COMPONENTS = [  # name, stellar mass (Msun), radial scale length (kpc), vertical scale height (kpc)
    ("thin disk", 3.5e10, 2.6, 0.30),
    ("thick disk", 6e9, 2.0, 0.90),
    ("bulge", 1.5e10, 0.55, 0.18),
]
N_MC = 2_000_000
THETAS_DEG = [1, 2, 3, 5, 10, 20, 30, 90]
REACH = [("3.6 m, R = 115,000", co.T_EXP), ("39 m, R = 100,000", co.T_EXP)]  # one exposure: the belt drifts


def galaxy(rng):
    """Positions (kpc) of N_MC stars, host-centred: distance and Galactic latitude."""
    x, y, z = stars(rng)
    dx, dz = x - R0_KPC, z - Z0_KPC
    rho = np.hypot(dx, y)
    return np.hypot(rho, dz), np.degrees(np.arctan2(np.abs(dz), rho))


def stars(rng, n_mc: int = N_MC):
    """Galactocentric positions x, y, z (kpc) of n_mc stars drawn from the model."""
    mass = np.array([c[1] for c in COMPONENTS])
    n = rng.multinomial(n_mc, mass / mass.sum())
    xs, ys, zs = [], [], []
    for (_, _, rd, hz), k in zip(COMPONENTS, n, strict=True):
        R = rng.gamma(2.0, rd, k)  # exponential surface density: p(R) ~ R exp(-R / rd)
        phi = rng.uniform(0, 2 * np.pi, k)
        xs.append(R * np.cos(phi))
        ys.append(R * np.sin(phi))
        zs.append(rng.laplace(0.0, hz, k))
    return tuple(np.concatenate(a) for a in (xs, ys, zs))


def density(x, y, z):
    """Stellar mass density (Msun kpc^-3) of the model at Galactocentric x, y, z (kpc)."""
    R = np.hypot(x, y)
    return sum(m / (2 * np.pi * rd**2) * np.exp(-R / rd) / (2 * hz) * np.exp(-np.abs(z) / hz)
               for _, m, rd, hz in COMPONENTS)


def local_cover(rng, d_max, thetas):
    """Fraction of the stars within d_max (kpc) of the host that lie at |b| < theta, by sampling the
    sphere uniformly and weighting by the model density (the global sample is too sparse nearby)."""
    n = 400_000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    r = d_max * rng.uniform(0, 1, n) ** (1 / 3)
    dx, dy, dz = (u * r[:, None]).T
    w = density(R0_KPC + dx, dy, Z0_KPC + dz)
    b = np.degrees(np.arctan2(np.abs(dz), np.hypot(dx, dy)))
    return {th: float(np.sum(w[b < th]) / np.sum(w)) for th in thetas}


def reach_kpc(inst, t_exp):
    """Distance (kpc) at which one exposure of a Sun-like star reaches S/N = 300 at 4297 A."""
    curve = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))["G"]["Tc I"]
    ln, fr = max(zip(curve.lines, curve.flux_ratio, strict=True), key=lambda t: t[0].loggf)
    mv = survey.absolute_v(co.TEFF)
    f = lambda x: np.log10(detect.snr_per_pixel(inst, mv + 5 * (x - 1), t_exp, ln.wl, fr) / co.SNR)  # noqa: E731
    return 10 ** brentq(f, -2, 6) / 1e3


def zone(sp, inst, r0_au, beta_val, tau1):
    """Cheapest release from rest at r0: column multiple X, visibility time, and cost per steradian."""
    r0 = r0_au * AU
    rs = co.R_STAR * RSUN
    t_exc = co.TEFF if 0.25 * (rs / r0) ** 2 > 1e-3 else 300.0
    b = beta_val[t_exc]
    gm = comet.G * co.M_STAR * MSUN * (b - 1.0)
    dv = max(C / 1e5 / inst.R, (b - 1) * comet.G * co.M_STAR * MSUN / r0**2 * co.T_EXP / 1e5)
    n_thr = comet.threshold(sp, inst, co.SNR, alpha=1.0, dv_kms=dv, t_exc=t_exc)
    y = np.linspace(1.0, 40.0, 40001)
    t = np.sqrt(r0**3 / (2 * gm)) * (np.sqrt(y * (y - 1)) + np.log(np.sqrt(y) + np.sqrt(y - 1)))
    rate = 1.0 / (tau1 * y**2 * r0_au**2)  # photoionization rate at r = y r0
    lnS = -np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    x_need = y**2 * np.exp(-lnS)  # X that keeps the column at threshold until r = y r0
    cost = x_need[1:] * np.array([comet.event_rate(tt) for tt in t[1:]])  # power per unit X: mass times rate
    k = int(np.argmin(cost)) + 1
    X, t_vis = float(x_need[k]), float(t[k])
    sigma_g = X * n_thr * sp.mass * AMU  # surface density (g cm^-2)
    return dict(r0_au=r0_au, t_exc=t_exc, dv_kms=dv, n_thr=n_thr, X=X, y_end=float(y[k]), t_vis_s=t_vis,
                tau_max=comet.max_tau(sp, X * n_thr, 1.0, dv, t_exc), mass_4pi_g=sigma_g * 4 * np.pi * r0**2)


def cost(z, sin_theta):
    nuc = hosts.ROUTES[co.ROUTE][0]
    e_per_g = hosts.energy_per_atom(co.ROUTE) / (hosts.NUCLIDES[nuc]["A"] * AMU)
    m = z["mass_4pi_g"] * sin_theta
    f_events = comet.event_rate(z["t_vis_s"])
    p = f_events * m / YR * e_per_g / 1e7
    p_on = m / z["t_vis_s"] * e_per_g / 1e7
    return dict(mass_kg=m / 1e3, events_per_yr=f_events, power_W=p, sagan_K=(np.log10(p) - 6) / 10,
                power_on_W=p_on, sagan_K_on=(np.log10(p_on) - 6) / 10)


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def main():
    sp = comet.species()["Tc I"]
    insts = {i.name: i for i in detect.INSTRUMENTS}
    inst = insts[co.INSTRUMENT]
    wl_flux = co.flux_nu_factory()
    beta_val = {t: comet.beta(sp, wl_flux, co.M_STAR, co.R_STAR, t) for t in (co.TEFF, 300.0)}
    tau1 = 1.0 / co.PHOTOION_1AU[co.ANALOG]

    zones = [zone(sp, inst, float(r), beta_val, tau1) for r in np.logspace(np.log10(0.05), 1.0, 61)]
    best = min(zones, key=lambda z: z["mass_4pi_g"] / z["t_vis_s"])
    print(f"best release r0 = {best['r0_au']:.3f} AU: X = {best['X']:.2f}, visible {best['t_vis_s'] / 86400:.2f} d "
          f"until r = {best['y_end']:.2f} r0, dv = {best['dv_kms']:.1f} km/s, tau_max = {best['tau_max']:.2f}")
    sens = {}
    for analog in ("Fe", "Mn", "Co"):
        zs = [zone(sp, inst, float(r), beta_val, 1.0 / co.PHOTOION_1AU[analog]) for r in np.logspace(np.log10(0.05), 1.0, 61)]
        b = min(zs, key=lambda z: z["mass_4pi_g"] / z["t_vis_s"])
        sens[analog] = dict(r0_au=b["r0_au"], sagan_K_iso=cost(b, 1.0)["sagan_K"])
        print(f"  analog {analog}: r0 = {b['r0_au']:.3f} AU, K(isotropic) = {sens[analog]['sagan_K_iso']:.2f}")

    rng = np.random.default_rng(1)
    _, blat = galaxy(rng)
    reach = {name: reach_kpc(insts[name], t) for name, t in REACH}
    for name, dk in reach.items():
        print(f"S/N = 300 in one exposure at 4297 A for a Sun-like star: {name} out to {dk * 1e3:.0f} pc")

    local = {name: local_cover(rng, reach[name], THETAS_DEG + [1.30]) for name, _ in REACH}
    rows = []
    for th in THETAS_DEG:
        # observers at |b| < th see the whole disc only if the zone extends R*/r0 beyond them
        def zone_cost(z, th=th):
            half = min(90.0, th + np.degrees(co.R_STAR * RSUN / (z["r0_au"] * AU)))
            return z["mass_4pi_g"] * np.sin(np.radians(half)) / z["t_vis_s"], half
        z = min(zones, key=lambda z: zone_cost(z)[0])
        half_zone = zone_cost(z)[1]
        c = cost(z, np.sin(np.radians(half_zone)))
        c["transport_dv_kms"] = co.transport_dv("at rest", z["r0_au"] * AU) / 1e5
        cover = {"all": float(np.mean(blat < th)), **{name: local[name][th] for name in local}}
        rows.append(dict(theta_deg=th, zone_half_deg=half_zone, r0_au=z["r0_au"], t_vis_s=z["t_vis_s"],
                         sky_fraction=float(np.sin(np.radians(th))), cover=cover, **c))
        print(f"Theta = {th:5.1f} deg (zone {half_zone:5.2f}, r0 = {z['r0_au']:.3f} AU, visible "
              f"{z['t_vis_s'] / 86400:.2f} d): sky {np.sin(np.radians(th)):.4f}, stars "
              + ", ".join(f"{k}: {v:.3f}" for k, v in cover.items())
              + f"; M = {c['mass_kg']:.1e} kg, K = {c['sagan_K']:.2f}, K_on = {c['sagan_K_on']:.2f}")
    half = brentq(lambda th: np.mean(blat < th) - 0.5, 0.01, 90)
    print(f"half of the Galaxy's stars lie within |b| < {half:.2f} deg; within the 39 m reach "
          f"{local['39 m, R = 100,000'][1.30]:.3f} of stars lie there")

    targeted = json.loads((ROOT / "research" / "comet_opt.json").read_text())["best"]
    r39, r36 = reach["39 m, R = 100,000"], reach["3.6 m, R = 115,000"]
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Beacon & $r_0$ (AU) & Sky & \multicolumn{3}{c}{Stars covered} & Mass (kg) & $K$ \\",
           r"\cmidrule(lr){4-6}",
           rf" & & & All & $<{r39 * 1e3:.0f}$~pc & $<{r36 * 1e3:.0f}$~pc & & \\",
           r"\midrule",
           f"Targeted cloud & {targeted['r_au']:.1f} & {sci(0.25 * (RSUN / (targeted['r_au'] * AU)) ** 2)} & \\multicolumn{{3}}{{c}}{{one system}} & "
           f"{sci(targeted['m_event_g'] / 1e3)} & {targeted['sagan_K']:.2f} \\\\"]
    for r in rows:
        lab = "Isotropic shell" if r["theta_deg"] == 90 else rf"Belt, $|b| < {r['theta_deg']:g}^\circ$"
        if r["theta_deg"] in (1, 3, 10, 30, 90):
            tex.append(f"{lab} & {r['r0_au']:.2f} & {r['sky_fraction']:.3f} & {r['cover']['all']:.2f} & "
                       f"{r['cover']['39 m, R = 100,000']:.2f} & {r['cover']['3.6 m, R = 115,000']:.2f} & "
                       f"{sci(r['mass_kg'])} & {r['sagan_K']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "broadcast.tex").write_text("\n".join(tex) + "\n")
    out = dict(best_zone=best, beta=beta_val, sensitivity=sens, reach_kpc=reach, half_theta_deg=half,
               components=COMPONENTS, rows=rows, zones=zones)
    (ROOT / "research" / "broadcast.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
