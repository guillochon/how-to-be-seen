"""Optimize a doped-gas beacon around a Sun-like star: release mode and release distance.

Writes research/comet_opt.json and paper/tables/comet_opt.tex.

Cost metric (as in scripts/comet.py): the fission power, for 99Tc bred from 98Mo, that supplies
the events at the rate that gives a monitoring program a probability of catching at least one
(beacon/comet.py, event_rate). An event is one cloud of dopant at the 5-sigma column, kept optically thin.

Only neutral Tc is used. Around a Sun-like star, Tc+ is picked up by the magnetized stellar wind
within about a gyroperiod and spread over a shell of radius equal to the wind speed (Gaffey et al.
1988), hundreds of km/s (Woch et al. 1997), so it cannot hold a narrow absorption line.

Release modes (all clouds released at the transit):
  periastron  released by a body on a parabolic orbit at its periastron; crossing time at that speed
  orbit       released from a circular orbit at radius r; crossing time at the orbital speed
  at rest     released at rest on the line of sight (apoapsis of a radial orbit, or a statite); it
              moves radially under radiation pressure and stays on the disc until its distance has
              doubled
The residence is the smallest of the crossing time and the lateral spreading time (release
velocity dispersion at T_REL plus the random walk of photon recoils), and is then reduced by
photoionization: t_eff = tau_ion (1 - exp(-t_res / tau_ion)), with tau_ion scaling as r^2 from its
value at 1 AU for a neutral analog atom (Heays et al. 2017; Mn, the group-7 homologue of Tc).
Excitation: Boltzmann at Teff within 16 R_star, ground level only beyond.
Velocity window: the instrumental width c/R or the drift |beta - 1| g t_exp in one exposure.

    uv run python scripts/comet_opt.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, atlas, atoms, comet, detect, hosts, survey, synth  # noqa: E402
from beacon.const import AMU, AU, MSUN, RSUN, YR, C, H, K  # noqa: E402

M_STAR, R_STAR, TEFF = 1.0, 1.0, 5777.0
SPECTRA_PER_YR = 9000 / 17.0
SNR = 300.0
T_EXP = 900.0  # s, one exposure (assumed)
T_REL = 100.0  # K, temperature of the release velocity dispersion (fiducial; scanned below)
B_FRAC = 0.5
ROUTE = "99Tc capture"
G = comet.G
V_1AU = np.sqrt(G * MSUN / AU)
INSTRUMENT = "3.6 m, R = 115,000"
# Solar-field photoionization rates of neutral atoms (Heays et al. 2017, Table, "Solar" column,
# normalized to the Draine field) times their factor 37,700 for the Sun at 1 AU.
PHOTOION_1AU = {"Fe": 3.9e-11 * 37700, "Mn": 5.8e-12 * 37700, "Co": 3.5e-12 * 37700}
ANALOG = "Mn"
LAMBDA_REC = 4297e-8  # cm, the strongest Tc I resonance line, for the photon recoil


def flux_nu_factory():
    s = synth.Synth(atlas.sun())
    cache = {}

    def f(wl):
        key = round(wl, 1)
        if key not in cache:
            _, _, F = s.profile(atoms.Line(wl, -9.0, 0.0, 26, 0), -20.0, npts=3)
            cache[key] = F * (wl * 1e-8) ** 2 / C
        return cache[key]
    return f


def transport_dv(mode: str, r_cm: float) -> float:
    """Delta-v (cm/s) from a circular orbit at 1 AU to the release state (impulsive, two burns)."""
    gm = G * M_STAR * MSUN
    a_t = 0.5 * (AU + r_cm)
    v_dep = np.sqrt(gm * (2 / AU - 1 / a_t))
    v_arr = np.sqrt(gm * (2 / r_cm - 1 / a_t))
    dv1 = abs(v_dep - V_1AU)
    if mode == "orbit":
        return dv1 + abs(np.sqrt(gm / r_cm) - v_arr)
    if mode == "at rest":
        return dv1 + v_arr
    return dv1


def spreading_time(mass_u: float, t_rel: float, beta_val: float, r: float) -> float:
    """Time (s) for the cloud to spread laterally by one stellar radius.

    Lateral speed: v^2(t) = v0^2 + (2/3) v_rec^2 Gamma t, with v0 the release dispersion at t_rel,
    v_rec = h / (lambda m) the recoil of one scattered photon, and Gamma = beta g / v_rec the
    scattering rate implied by the radiation acceleration; isotropic re-emission random-walks the
    velocity across the line of sight.
    """
    m = mass_u * AMU
    v0 = np.sqrt(2 * K * t_rel / m)
    v_rec = H / (LAMBDA_REC * m)
    gam = beta_val * G * M_STAR * MSUN / r**2 / v_rec
    rs = R_STAR * RSUN
    f = lambda t: np.sqrt(v0**2 + 2 / 3 * v_rec**2 * gam * t) * t - rs  # noqa: E731
    return float(brentq(f, 1.0, 1e14))


def design(sp, inst, mode, r_au, beta_val, t_rel=T_REL, analog=ANALOG):
    rs = R_STAR * RSUN
    r = r_au * AU
    gm = G * M_STAR * MSUN
    g = gm / r**2
    t_exc = TEFF if 0.25 * (rs / r) ** 2 > 1e-3 else 300.0
    if mode == "periastron":
        t_cross = 2 * rs / np.sqrt(2 * gm / r)
    elif mode == "orbit":
        t_cross = 2 * rs / np.sqrt(gm / r)
    else:
        t_cross = comet.residence_time(r_au, beta_val, M_STAR, B_FRAC)
    t_spread = spreading_time(sp.mass, t_rel, beta_val, r)
    t_res = min(t_cross, t_spread)
    tau_ion = r_au**2 / PHOTOION_1AU[analog]
    t_eff = tau_ion * -np.expm1(-t_res / tau_ion)
    dv = max(C / 1e5 / inst.R, abs(beta_val - 1) * g * T_EXP / 1e5)
    alpha = 0.4
    for _ in range(30):
        n = comet.threshold(sp, inst, SNR, alpha=alpha, dv_kms=dv, t_exc=t_exc)
        if comet.max_tau(sp, n, alpha, dv, t_exc) <= 1.0 or alpha >= 1.0:
            break
        alpha = min(1.0, alpha * 1.25)
    m_event = comet.column_mass(n, sp.mass, R_STAR, alpha)
    f_events = comet.event_rate(t_eff)
    nuc = hosts.ROUTES[ROUTE][0]
    e_per_g = hosts.energy_per_atom(ROUTE) / (hosts.NUCLIDES[nuc]["A"] * AMU)
    p = f_events * m_event / YR * e_per_g / 1e7
    p_on = m_event / t_eff * e_per_g / 1e7
    dvt = transport_dv(mode, r)
    return dict(mode=mode, r_au=r_au, t_exc=t_exc, t_cross_s=t_cross, t_spread_s=t_spread, tau_ion_s=tau_ion,
                t_eff_s=t_eff, dv_kms=dv, alpha=alpha, n_thr=n, m_event_g=m_event, events_per_yr=f_events,
                power_W=p, sagan_K=(np.log10(p) - 6) / 10, power_on_W=p_on, sagan_K_on=(np.log10(p_on) - 6) / 10,
                transport_dv_kms=dvt / 1e5, transport_energy_over_production=0.5 * dvt**2 / e_per_g,
                t_rel=t_rel, analog=analog)


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def fmt_time(t):
    if t < 2 * 86400:
        return f"{t / 3600:.1f} h"
    if t < 2 * YR:
        return f"{t / 86400:.0f} d"
    return f"{t / YR:.0f} yr"


def optimum(sp, inst, beta_val, t_rel, analog):
    """Lowest-K design; beyond a few AU K is flat, so return the nearest release within 0.005 of it."""
    r_grid = np.logspace(np.log10(10 * RSUN / AU), 2, 41)
    designs = []
    for mode in ("periastron", "orbit", "at rest"):
        for r_au in r_grid:
            designs.append(design(sp, inst, mode, float(r_au),
                                  beta_val[TEFF if 0.25 * (RSUN / (r_au * AU)) ** 2 > 1e-3 else 300.0], t_rel, analog))
    k_min = min(d["sagan_K"] for d in designs)
    return min((d for d in designs if d["sagan_K"] <= k_min + 0.005), key=lambda d: d["r_au"])


def main():
    sp = comet.species()["Tc I"]
    inst = {i.name: i for i in detect.INSTRUMENTS}[INSTRUMENT]
    wl_flux = flux_nu_factory()
    beta_val = {t: comet.beta(sp, wl_flux, M_STAR, R_STAR, t) for t in (TEFF, 300.0)}
    print("beta(Tc I):", beta_val)
    out = dict(beta=beta_val, t_exp=T_EXP, t_rel=T_REL, analog=ANALOG, photoion_1au=PHOTOION_1AU,
               spectra_per_yr=SPECTRA_PER_YR, rows={}, sensitivity={})
    best = optimum(sp, inst, beta_val, T_REL, ANALOG)
    out["best"] = best
    print(f"optimum: {best['mode']} r = {best['r_au']:.2f} AU, K = {best['sagan_K']:.2f}, "
          f"P = {best['power_W']:.1e} W, M = {best['m_event_g'] / 1e3:.1e} kg, t_eff = {fmt_time(best['t_eff_s'])}")
    for t_rel in (30.0, 100.0, 1000.0, 3000.0):
        for analog in ("Fe", "Mn", "Co"):
            b = optimum(sp, inst, beta_val, t_rel, analog)
            out["sensitivity"][f"{t_rel:.0f}|{analog}"] = dict(mode=b["mode"], r_au=b["r_au"], sagan_K=b["sagan_K"],
                                                                sagan_K_on=b["sagan_K_on"], t_eff_s=b["t_eff_s"])
            print(f"  T_rel = {t_rel:6.0f} K, analog {analog}: {b['mode']:9s} r = {b['r_au']:6.2f} AU "
                  f"K = {b['sagan_K']:.2f} K_on = {b['sagan_K_on']:.2f} t_eff = {fmt_time(b['t_eff_s'])}")

    # Distance at which one exposure of a Sun-like star reaches S/N = 300 at the strongest Tc I line
    curve = pickle.load((ROOT / "research" / "curves.pkl").open("rb"))["G"]["Tc I"]
    ln, fr = max(zip(curve.lines, curve.flux_ratio, strict=True), key=lambda t: t[0].loggf)
    mv = survey.absolute_v(TEFF)
    logd = brentq(lambda x: np.log10(detect.snr_per_pixel(inst, mv + 5 * (x - 1), T_EXP, ln.wl, fr) / SNR), -2, 5)
    out["reach"] = dict(d_pc=10**logd, V=mv + 5 * (logd - 1), wl=ln.wl, t_exp=T_EXP)
    print(f"S/N = {SNR:.0f} in {T_EXP:.0f} s at {ln.wl:.0f} A out to {10**logd:.1f} pc (V = {out['reach']['V']:.2f})")

    # K against release distance, for paper/figures/make_artificial_comet.py
    r_grid = np.logspace(np.log10(10 * RSUN / AU), 2, 61)
    out["grid"] = dict(r_au=list(r_grid))
    for mode, t_rels in (("orbit", (T_REL,)), ("at rest", (30.0, T_REL, 1000.0))):
        for t_rel in t_rels:
            out["grid"][f"{mode}|{t_rel:.0f}"] = [
                design(sp, inst, mode, float(r_au),
                       beta_val[TEFF if 0.25 * (RSUN / (r_au * AU)) ** 2 > 1e-3 else 300.0], t_rel)["sagan_K"]
                for r_au in r_grid]

    rows = [("Periastron, 10 $R_\\odot$", "periastron", 10 * RSUN / AU), ("Orbit, 1 AU", "orbit", 1.0),
            ("Orbit, 30 AU", "orbit", 30.0), ("At rest, 0.2 AU", "at rest", 0.2),
            ("At rest, 1 AU", "at rest", 1.0), ("At rest, 3 AU", "at rest", 3.0), ("At rest, 10 AU", "at rest", 10.0)]
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Release & $t_\mathrm{cross}$ & $t_\mathrm{spread}$ & $\tau_\mathrm{ion}$ & $\Delta v$ & Mass (kg) & $K$ & $K_\mathrm{on}$ \\",
           r"\midrule"]
    for lab, mode, r_au in rows:
        b = beta_val[TEFF if 0.25 * (RSUN / (r_au * AU)) ** 2 > 1e-3 else 300.0]
        d = design(sp, inst, mode, r_au, b)
        out["rows"][lab] = d
        tex.append(f"{lab} & {fmt_time(d['t_cross_s'])} & {fmt_time(d['t_spread_s'])} & {fmt_time(d['tau_ion_s'])} & "
                   f"{d['dv_kms']:.1f} & {sci(d['m_event_g'] / 1e3)} & {d['sagan_K']:.2f} & {d['sagan_K_on']:.2f} \\\\")
        print(f"{lab:28s} cross={fmt_time(d['t_cross_s']):>7s} spread={fmt_time(d['t_spread_s']):>6s} "
              f"ion={fmt_time(d['tau_ion_s']):>7s} t_eff={fmt_time(d['t_eff_s']):>6s} dv={d['dv_kms']:5.1f} "
              f"M={d['m_event_g'] / 1e3:.1e} K={d['sagan_K']:.2f} K_on={d['sagan_K_on']:.2f} "
              f"transport={d['transport_dv_kms']:.1f} km/s ({d['transport_energy_over_production']:.0e})")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "comet_opt.tex").write_text("\n".join(tex) + "\n")
    (ROOT / "research" / "comet_opt.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
