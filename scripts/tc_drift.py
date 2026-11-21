"""How fast radiation drives Tc up through the atmosphere of the quiet late-B host, and what that does to
the upkeep if nothing holds the Tc at the top.

A trace ion of mass A m_u and charge Z drifts at v = D F / kT, with F = A m_u (g_rad - g) + Z m_p g / 2 (the
second term is the electric field of an ionized hydrogen atmosphere). D combines collisions with protons
(Chapman-Cowling first approximation for a Coulomb potential, D = 3 (kT)^5/2 / (4 sqrt(2 pi mu) n_p Z^2
e^4 ln Lambda), ln Lambda = ln(lambda_Debye / (Z e^2 / kT))) and, for neutral Tc, the polarization
(Langevin) rate on neutral hydrogen. g_rad per ion is from research/levitation.json at the threshold of the
Tc II space beacon. Helium is ignored, which overestimates D slightly.

If the top of the atmosphere is open (any wind removes what arrives there), a steady beacon must be
supplied at the rate that the Tc flows through it: the column of Tc is N = F t_res, with
t_res = int dz / v from tau5000 = 3 to the top. If it is closed, Tc collects at the top instead. A
horizontal magnetic field could close it: b_trap is the field above which Tc ions gyrate faster than they
collide, so that the field slows their vertical drift. held_layer compares the Tc II lines of Tc held in
such a high layer with those of the same column spread through the atmosphere.

    uv run python scripts/tc_drift.py     # writes research/tc_drift.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, atlas, atoms, hosts  # noqa: E402
from beacon import levitation as L  # noqa: E402
from beacon.const import AMU, K  # noqa: E402
from beacon.synth import Synth  # noqa: E402

YR = 3.156e7
E = 4.803e-10
MP = 1.6726e-24
C_LIGHT = 2.998e10
ALPHA_H = 0.667e-24  # cm^3, polarizability of H
A_TC = 98.906
TAU_BASE = 3.0
G_CGS, MSUN = 6.674e-8, 1.989e33


def coulomb_D(T, n_p, n_e, z):
    mu = A_TC * 1.008 / (A_TC + 1.008) * AMU
    kT = K * T
    lam_d = np.sqrt(kT / (4 * np.pi * n_e * E**2))
    lnL = np.log(np.maximum(lam_d / (z * E**2 / kT), 10.0))
    return 3 * kT**2.5 / (4 * np.sqrt(2 * np.pi * mu) * n_p * z**2 * E**4 * lnL)


def neutral_D(T, n_h0):
    mu = A_TC * 1.008 / (A_TC + 1.008) * AMU
    k_l = 1.105 * 2 * np.pi * E * np.sqrt(ALPHA_H / mu)  # momentum-transfer rate coefficient
    return K * T / (mu * n_h0 * k_l)


def drift(teff, logg, rec, tau_max=TAU_BASE):
    """Depth profile above tau_max: (log tau, rho, z below the top, drift speed v, D per ion), with g_rad
    per ion from the levitation record rec (held flat beyond its depth range)."""
    atm = L.Atmosphere(teff, logg, 0.5)
    g = 10**logg
    lt = np.log10(atm.tau5)
    sel = lt <= np.log10(tau_max)
    T, ne, rho, rhox = atm.T[sel], atm.ne[sel], atm.rho[sel], atm.rhox[sel]
    n_p, n_h0 = atm.s.st.nHII[sel], atm.s.st.nHI[sel]
    frac = atm.frac(43)[:, sel]
    lg_ion = np.array(rec["log_grad_ion"])
    grad = np.array([10 ** np.interp(lt[sel], rec["taus"], lg_ion[i]) for i in range(3)])
    D = np.array([neutral_D(T, n_h0),
                  1 / (1 / coulomb_D(T, n_p, ne, 1) + 1 / neutral_D(T, n_h0)),
                  1 / (1 / coulomb_D(T, n_p, ne, 2) + 1 / neutral_D(T, n_h0))])
    F = np.array([A_TC * AMU * (grad[i] - g) + i * MP * g / 2 for i in range(3)])
    # the element drifts at the fraction-weighted mean (ions exchange charge far faster than they drift)
    v = (frac * D * F / (K * T)).sum(axis=0)
    z = np.concatenate([[0.0], np.cumsum(np.diff(rhox) / (0.5 * (rho[1:] + rho[:-1])))])
    return lt[sel], T, rho, z, v, D


def leak_time(h, rec, tau_top):
    """Mean time for Tc to leak out of a mixed layer whose top lies at tau_top, through the unmixed
    layer above it: M_mix / (4 pi R^2 rho v) at tau_top, in years."""
    lt, _T, rho, _z, v, _D = drift(h.teff, h.logg, rec, 10.0)
    flux = np.interp(np.log10(tau_top), lt, rho * v)  # g cm^-2 s^-1 of gas through which Tc passes
    r2 = G_CGS * h.mass * MSUN / 10**h.logg
    return float(h.mmix * MSUN / (4 * np.pi * r2 * flux) / YR)


def held_layer(teff, logg, log_eps, tops=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), nlines=6):
    """Equivalent widths of the strongest Tc II ultraviolet lines when the Tc of a uniform layer above
    tau5000 = 3 is held instead above tau5000 = top at the same column, as ratios to the uniform case."""
    s = Synth(atlas.model(teff, logg), vturb=1.0)
    t5, rx = s.tau_cont_500(), s.m.rhox
    lines = sorted(atoms.kurucz_lines("gf4301.pos", 43, 1, 1900.0, 3300.0), key=lambda ln: -ln.loggf)[:nlines]
    col = lambda m: np.trapezoid(m.astype(float), rx)  # noqa: E731
    out = {}
    for top in tops:
        m = t5 < top
        le = log_eps + np.log10(col(t5 < TAU_BASE) / col(m))
        prof = np.where(m, le, -20.0)
        out[f"{top:g}"] = dict(log_eps_in_layer=float(le), ew_ratio={f"{ln.wl:.1f}": s.ew(ln, prof) / s.ew(ln, log_eps)
                                                                     for ln in lines})
    return out


def main():
    lev = json.loads((ROOT / "research" / "levitation.json").read_text())
    q = json.loads((ROOT / "research" / "quiet_host.json").read_text())
    rec = lev[f"tc|{q['teff']}|+0"]
    lt, T, rho, z, v, D = drift(q["teff"], q["logg"], rec)
    # A horizontal field slows the vertical drift of an ion by 1 / (1 + (omega_c / nu)^2). The field at which
    # the gyrofrequency Z e B / (m c) equals the momentum-transfer rate kT / (m D), the larger of Tc II and III:
    b_trap = np.maximum(K * T * C_LIGHT / (1 * E * D[1]), K * T * C_LIGHT / (2 * E * D[2]))
    t_cross = np.concatenate([[0.0], np.cumsum(np.diff(z) / (0.5 * (v[1:] + v[:-1])))]) / YR  # from the top down
    t_mean = hosts.NUCLIDES["99Tc"]["t_half"] / np.log(2)
    t_res = float(t_cross[-1])
    base = q["cases"]["no wind"]
    p_open = base["upkeep_W"] * (1 + t_mean / t_res)
    grid = {h.key: h for h in hosts.host_grid()}
    leaks = {}
    for key, teff in (("Am", 8000), ("A0", 9500)):
        r = lev[f"tc|{teff}|+0"] if f"tc|{teff}|+0" in lev else lev["tc|9000|+0"]
        leaks[key] = {f"{t:g}": leak_time(grid[key], r, t) for t in (0.1, 1.0, 3.0)}
    out = dict(
        teff=q["teff"], logg=q["logg"], log_eps=rec["log_eps"],
        depth=[dict(log_tau=float(x), T=float(t), D_TcII=float(d1), D_TcIII=float(d2), v_cm_s=float(vv),
                    t_from_top_yr=float(tc), b_trap_G=float(b)) for x, t, d1, d2, vv, tc, b in
               zip(lt[::6], T[::6], D[1][::6], D[2][::6], v[::6], t_cross[::6], b_trap[::6], strict=True)],
        height_km=float(z[-1] / 1e5), t_res_yr=t_res, t_mean_life_yr=float(t_mean),
        open_top=dict(upkeep_W=p_open, sagan_K=(np.log10(p_open) - 6) / 10, factor=float(1 + t_mean / t_res)),
        closed_top=dict(upkeep_W=base["upkeep_W"], sagan_K=base["sagan_K"]),
        wind_only=q["cases"]["wind"],
        a_star_leak_yr=leaks,
        held_layer=held_layer(q["teff"], q["logg"], rec["log_eps"]),
    )
    (ROOT / "research" / "tc_drift.json").write_text(json.dumps(out, indent=1))
    for r in out["depth"]:
        print(f"log tau {r['log_tau']:6.2f}  T {r['T']:6.0f}  D(II) {r['D_TcII']:.2e}  D(III) {r['D_TcIII']:.2e}  "
              f"v {r['v_cm_s']:.2e} cm/s  t(top->here) {r['t_from_top_yr']:.1f} yr  B_trap {r['b_trap_G']:.2g} G")
    print(f"height {out['height_km']:.0f} km, residence {t_res:.0f} yr, mean life {t_mean:.3g} yr")
    print(f"open top: upkeep x{out['open_top']['factor']:.3g} -> {p_open:.2e} W, K = {out['open_top']['sagan_K']:.2f}")
    for top, r in out["held_layer"].items():
        print(f"Tc held above tau {top}: log eps {r['log_eps_in_layer']:.2f}, EW / uniform",
              " ".join(f"{x:.1f}" for x in r["ew_ratio"].values()))
    print("A-star leak times (yr) by tau at the top of the mixed layer:", json.dumps(leaks))


if __name__ == "__main__":
    main()
