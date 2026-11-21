"""How long Tc stays in the turbulently mixed layer of the Am host, if radiation did not support it at all.

Within the mixed layer turbulence keeps the Tc uniform; it leaves through the bottom, at log(Delta M / M_*) = -5.5,
where turbulence gives way to atomic diffusion. The worst case for the beacon is g_rad = 0 there. A trace ion of
mass A m_u and charge Z then settles at v = D F / kT, with F = (A - Z / 2) m_u g (gravity less the electric field of
an ionized hydrogen plasma) and D the Chapman-Cowling coefficient for collisions with protons (scripts/tc_drift.py).
The mixed layer, of column mass m_mix = Delta M / (4 pi R^2), empties on the time scale
    t_leak = m_mix / (rho v)
at its base.

The conditions at the base: the envelope is thin, so g is the surface gravity and the pressure is P = g m_mix.
Richard, Michaud & Richer (2001) find T = 200,000 K (log T = 5.3) at log(Delta M / M_*) ~ -6.5 in A stars of
1.5-2.5 Msun; deeper, T rises as m^(1/4) to m^(1/3) (radiative envelope with constant to Kramers opacity), which puts
log T = 5.55-5.6 at -5.5. We bracket it with log T = 5.3-6.0 and take H and He fully ionized (mu = 0.62, Y = 0.27).
The charge of Tc at these temperatures is not tabulated; we bracket it with Z = 6-16, the range over which its
4d, 5s and 4p electrons are removed. D falls as Z^-2, so the lowest charge gives the fastest settling.

    uv run python scripts/am_settling.py      # writes research/am_settling.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tc_drift import A_TC, G_CGS, MSUN, YR, coulomb_D  # noqa: E402

from beacon import ROOT, hosts  # noqa: E402
from beacon.const import AMU, K  # noqa: E402

MU, Y = 0.62, 0.27
LOG_DM = [-6.0, -5.5, -5.0]
LOG_T = np.arange(5.3, 6.01, 0.1)
ZS = np.arange(6, 17, 2)
T_HALF = hosts.NUCLIDES["99Tc"]["t_half"]


def main():
    h = next(x for x in hosts.host_grid() if x.key == "Am")
    g = 10**h.logg
    R = np.sqrt(G_CGS * h.mass * MSUN / g)
    out = dict(host=h.key, mass_msun=h.mass, logg=h.logg, R_cm=R, t_half_yr=T_HALF, cases=[])
    for ldm in LOG_DM:
        m_mix = 10**ldm * h.mass * MSUN / (4 * np.pi * R**2)  # g cm^-2
        P = g * m_mix
        for lt in LOG_T:
            T = 10**lt
            rho = P * MU * AMU / (K * T)
            n_p = rho * (1 - Y) / AMU  # protons and, with He, electrons
            n_e = n_p + 2 * rho * Y / (4 * AMU)
            for z in ZS:
                D = coulomb_D(T, n_p, n_e, z)
                v = D * (A_TC - z / 2) * AMU * g / (K * T)
                t = m_mix / (rho * v) / YR
                out["cases"].append(dict(log_dm=ldm, log_T=float(lt), Z=int(z), m_mix_g_cm2=m_mix, rho=rho, D=D,
                                         v_cm_s=v, t_leak_yr=t))
    fid = [c for c in out["cases"] if c["log_dm"] == -5.5]
    tmin = min(fid, key=lambda c: c["t_leak_yr"])
    out["fiducial_min"] = tmin
    out["all_min"] = min(out["cases"], key=lambda c: c["t_leak_yr"])
    (ROOT / "research" / "am_settling.json").write_text(json.dumps(out, indent=1))
    for ldm in LOG_DM:
        cs = [c for c in out["cases"] if c["log_dm"] == ldm]
        lo = min(cs, key=lambda c: c["t_leak_yr"])
        mid = next(c for c in cs if abs(c["log_T"] - 5.6) < 0.01 and c["Z"] == 10)
        print(f"log dM/M = {ldm}: m_mix = {lo['m_mix_g_cm2']:.1e} g/cm^2; t_leak at log T 5.6, Z 10: "
              f"{mid['t_leak_yr']:.1e} yr (v = {mid['v_cm_s']:.1e} cm/s); shortest over the brackets "
              f"{lo['t_leak_yr']:.1e} yr (log T {lo['log_T']:.1f}, Z {lo['Z']}) = {lo['t_leak_yr'] / T_HALF:.0f} t_half")


if __name__ == "__main__":
    main()
