"""Absorption by a doped exocomet tail transiting its star, and the column a search can detect.

The model is that of Vrignaud et al. (2024a), introduced by Kiefer et al. (2014) for beta Pic:
a homogeneous cloud covers a fraction alpha of the stellar disc, and its gas is spread uniformly
over a radial-velocity interval Delta v. The mean absorption depth in a line (l, u) is
    abs = alpha * (1 - exp(-tau)),   tau = (pi e^2 / m_e c) * f * lambda * N_l / Delta v,
with the lower level populated in Boltzmann equilibrium at the excitation temperature T_exc
(radiative regime: T_exc close to the stellar Teff; Vrignaud & Lecavelier des Etangs 2024).
The dopant is assumed to be in its first ion, like Fe+, Cr+ and Ca+ in observed exocomets.

Detection: the mean depth over Delta v is measured on n_pix = Delta v / (c / (R * PIX_PER_RES))
pixels with per-pixel noise 1/(S/N), inflated by CONT_FACTOR for the reference spectrum, and the
significances of all covered lines add in quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import atoms, detect
from .const import AMU, LINE_CROSS, MSUN, RSUN, C

HC_K = atoms.HC_K

# The December 6, 1997 comet of beta Pic (Vrignaud et al. 2024a, b): covering factor, velocity
# range (+12 to +40 km/s), Fe II column and the measured Cr+/Fe+ and Ni+/Fe+ ratios.
ALPHA = 0.4
DELTA_V = 28.0  # km/s
N_FEII = 1.11e15  # cm^-2
CR_OVER_FE = 1.04e-2
NI_OVER_FE = 8.5e-2
T_EXC = 8000.0  # K, the Am host's Teff

# Monitoring. A gas beacon is costed by the rate of events that gives a monitoring program of T_PROGRAM years,
# taking spectra at random times, a probability P_DETECT of catching at least one event: with events at rate f
# and each caught with probability p = 1 - exp(-n t) (n spectra a year, t the time an event stays detectable),
# P = 1 - exp(-f T p). Ground-based: the HARPS cadence on beta Pic, more than 9000 spectra in 2003-2020
# (Hoeijmakers et al. 2025). Space ultraviolet: a cadence that one program could sustain on one star.
P_DETECT = 0.5
T_PROGRAM = 10.0  # yr
SPECTRA_PER_YR = 9000 / 17.0
SPECTRA_PER_YR_SPACE = 20.0


def event_rate(t_s: float, n_per_yr: float = SPECTRA_PER_YR) -> float:
    """Events per year that give probability P_DETECT of catching at least one in T_PROGRAM years."""
    p = -np.expm1(-n_per_yr * t_s / 3.15576e7)
    return float(-np.log1p(-P_DETECT) / (T_PROGRAM * p))


@dataclass
class Species:
    name: str
    z: int
    lines: list
    mass: float  # nuclide mass (u)
    ion: int = 1  # 0 neutral, 1 singly ionized


def species() -> dict[str, Species]:
    return {
        "Tc I": Species("Tc I", 43, [ln for ln in atoms.kurucz_lines("gf4300.pos", 43, 0, 2000, 10000)
                                      if ln.loggf >= -1.0], atoms.MASS[43], ion=0),
        "Tc II": Species("Tc II", 43, atoms.tc2_lines(min_loggf=-1.0), atoms.MASS[43]),
        "Pm II": Species("Pm II", 61, atoms.pm2_lines(), atoms.MASS[61]),
        "Ra II": Species("Ra II", 88, atoms.ra2_lines(), atoms.MASS[88]),
        "Cr II": Species("Cr II", 24, atoms.ion_lines("gf2401.all", 24, -1.0), atoms.MASS[24]),
        "Ca II": Species("Ca II", 20, atoms.ion_lines("gf2001.all", 20, -1.0), atoms.MASS[20]),
    }


def depths(sp: Species, n_tot: float, alpha: float = ALPHA, dv_kms: float = DELTA_V, t_exc: float = T_EXC):
    """Mean absorption depth of every line of a species for a total ion column n_tot (cm^-2)."""
    U = float(atoms.partition(sp.z, sp.ion, np.array([t_exc]))[0])
    out = []
    for ln in sp.lines:
        nl_over_g = n_tot * np.exp(-HC_K * ln.elow / t_exc) / U
        tau = LINE_CROSS * 10.0**ln.loggf * nl_over_g * ln.wl * 1e-8 / (dv_kms * 1e5)
        out.append(alpha * -np.expm1(-tau))
    return np.array(out)


def significance(sp: Species, n_tot: float, inst: detect.Instrument, snr: float,
                 alpha: float = ALPHA, dv_kms: float = DELTA_V, t_exc: float = T_EXC) -> float:
    d = depths(sp, n_tot, alpha, dv_kms, t_exc)
    n_pix = dv_kms / (C / 1e5 / (inst.R * detect.PIX_PER_RES))
    sigma = detect.CONT_FACTOR / (snr * np.sqrt(n_pix))
    sel = np.array([inst.wmin <= ln.wl <= inst.wmax for ln in sp.lines])
    if not sel.any():
        return 0.0
    return float(np.sqrt(np.sum((d[sel] / sigma) ** 2)))


def threshold(sp: Species, inst: detect.Instrument, snr: float, z_detect: float = 5.0, **kw) -> float:
    """Total ion column (cm^-2) that gives a combined significance z_detect; nan if no line is covered."""
    if significance(sp, 1e20, inst, snr, **kw) == 0.0:
        return float("nan")
    g = lambda logn: np.log10(max(significance(sp, 10**logn, inst, snr, **kw), 1e-30) / z_detect)  # noqa: E731
    return float(10 ** brentq(g, 4.0, 20.0, xtol=1e-3))


def best_lines(sp: Species, n_tot: float, inst: detect.Instrument, k: int = 3) -> list:
    d = depths(sp, n_tot)
    covered = [(dd, ln) for dd, ln in zip(d, sp.lines, strict=True) if inst.wmin <= ln.wl <= inst.wmax]
    return [ln for _, ln in sorted(covered, key=lambda t: -t[0])[:k]]


def column_mass(n_tot: float, mass_u: float, r_star_rsun: float, alpha: float = ALPHA) -> float:
    """Mass (g) of the ions in the part of the tail that covers the stellar disc."""
    return n_tot * alpha * np.pi * (r_star_rsun * RSUN) ** 2 * mass_u * AMU


G = 6.674e-8


def parabolic_time(q: float, r: float, m_star_msun: float) -> float:
    """Time (s) for a parabolic orbit to go from periastron q to distance r (cm)."""
    d = np.sqrt(r / q - 1.0)
    return np.sqrt(2 * q**3 / (G * m_star_msun * MSUN)) * (d + d**3 / 3.0)


def event_timing(m_star_msun: float, r_star_rsun: float, q_rstar: float, rsub_rstar: float) -> dict:
    """Transit duration at periastron and the time a comet spends inside its sublimation radius.

    The cloud crosses the stellar diameter at the periastron speed; the comet releases its
    refractory dopant for as long as it is inside rsub. eta_nat is the fraction of a continuous
    release that is in front of the star during the transit.
    """
    rs = r_star_rsun * RSUN
    q = q_rstar * rs
    v_q = np.sqrt(2 * G * m_star_msun * MSUN / q)
    t_tr = 2 * rs / v_q
    t_act = 2 * parabolic_time(q, rsub_rstar * rs, m_star_msun)
    return dict(v_q_kms=v_q / 1e5, t_transit_s=t_tr, t_active_s=t_act, eta_nat=t_tr / t_act)


__all__ = ["species", "depths", "significance", "threshold", "column_mass", "MSUN"]


def max_tau(sp: Species, n_tot: float, alpha: float = ALPHA, dv_kms: float = DELTA_V, t_exc: float = T_EXC) -> float:
    """Largest line-centre optical depth of a species for a total column n_tot."""
    U = float(atoms.partition(sp.z, sp.ion, np.array([t_exc]))[0])
    taus = [LINE_CROSS * 10.0**ln.loggf * n_tot * np.exp(-HC_K * ln.elow / t_exc) / U * ln.wl * 1e-8 / (dv_kms * 1e5)
            for ln in sp.lines]
    return float(max(taus))


def beta(sp: Species, wl_flux, m_star_msun: float, r_star_rsun: float, t_exc: float) -> float:
    """Radiation pressure over gravity for a species in Boltzmann equilibrium at t_exc.

    wl_flux(wl_A) returns the stellar surface flux density F_nu (erg s^-1 cm^-2 Hz^-1) at wl_A; both
    forces fall as r^-2, so the ratio is independent of distance (optically thin cloud).
    """
    U = float(atoms.partition(sp.z, sp.ion, np.array([t_exc]))[0])
    acc = 0.0
    for ln in sp.lines:
        frac_over_g = np.exp(-HC_K * ln.elow / t_exc) / U
        acc += LINE_CROSS * 10.0**ln.loggf * frac_over_g * wl_flux(ln.wl) / C / (sp.mass * AMU)
    acc *= (r_star_rsun * RSUN) ** 2
    return float(acc / (G * m_star_msun * MSUN))


def residence_time(r_au: float, beta_val: float, m_star_msun: float, b_frac: float = 0.5) -> float:
    """Time (s) a cloud released at rest on the line of sight, at r_au from the star, stays projected
    on the stellar disc. It moves radially (toward the observer if beta > 1, toward the star if
    beta < 1); a cloud at fractional impact parameter b_frac leaves the disc when its distance has
    grown by 1/b_frac. Inward motion keeps it on the disc until it reaches the star.
    """
    r0 = r_au * 1.495978707e13
    gm = G * m_star_msun * MSUN * abs(beta_val - 1.0)
    if beta_val < 1.0:  # free fall from rest to the star
        return float(np.pi / 2 * np.sqrt(r0**3 / (2 * gm)))
    # Outward from rest under a = gm / r^2: v^2 = 2 gm (1/r0 - 1/r), and with y = r / r0
    # t(y) = sqrt(r0^3 / 2 gm) * [sqrt(y (y - 1)) + ln(sqrt(y) + sqrt(y - 1))].
    y = 1.0 / b_frac
    return float(np.sqrt(r0**3 / (2 * gm)) * (np.sqrt(y * (y - 1)) + np.log(np.sqrt(y) + np.sqrt(y - 1))))
