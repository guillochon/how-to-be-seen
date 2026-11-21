"""Host stars, the mass that dilutes a surface pollutant, and what the pollutant costs.

Mixed masses:
    FGK dwarfs   the surface convection zone, from Pinsonneault, DePoy & Coffee (2001, Fig. 1)
    M dwarfs     the whole star, for stars below the fully convective boundary
    A/Am stars   a turbulently mixed outer layer of log(Delta M / M_*) = -6 to -5, the range
                 that reproduces AmFm abundance anomalies (Vick et al. 2010); the H and He
                 convection zones alone reach only log(Delta M / M_*) ~ -10.
    late-B star  a quiet, slowly rotating (HgMn-like) star that mixes nothing below its atmosphere:
                 the mass above tau5000 = 3 (research/quiet_host.json, scripts/quiet_host.py).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np

from . import DATA, ROOT
from .const import AMU, EV, MEARTH, MSUN, YR

X_HYDROGEN = 0.7381  # present-day solar hydrogen mass fraction (Asplund et al. 2009)
M_H = 1.00794  # hydrogen atomic mass (u)


def pinsonneault_mcz(teff: float) -> float:
    """Surface convection zone mass (Msun) of a main-sequence star (4000 K <= Teff <= 6400 K)."""
    rows = [r for r in csv.reader(ln for ln in (DATA / "pinsonneault2001_cz.csv").open() if not ln.startswith("#"))]
    tab = np.array([[float(x) for x in r] for r in rows[1:]])
    if not tab[0, 0] <= teff <= tab[-1, 0]:
        raise ValueError(f"Teff={teff} outside 4000-6400 K")
    return 10.0 ** np.interp(teff, tab[:, 0], tab[:, 1])


@dataclass
class Host:
    key: str
    label: str
    teff: float
    logg: float
    mass: float  # stellar mass (Msun)
    mmix: float  # fiducial mixed mass (Msun)
    mmix_lo: float
    mmix_hi: float
    vturb: float  # microturbulence (km s^-1)
    vsini: float  # projected rotation (km s^-1) for the line-width estimate
    macro: float  # macroturbulent broadening FWHM (km s^-1)
    solar: bool = False


def _a_star_mix(mstar: float) -> tuple[float, float, float]:
    """(fiducial, low, high) mixed mass of an A star: log(DeltaM/M*) = -5.5, -10 (H and He
    convection zones only) and -5 (deepest turbulent mixing that fits AmFm stars; Vick et al. 2010)."""
    return mstar * 10**-5.5, mstar * 1e-10, mstar * 1e-5


def _quiet_b() -> Host:
    """The quiet late-B host of scripts/quiet_host.py, with the line broadening of its threshold in
    scripts/levitation.py (microturbulence 1 km/s, v sin i 10 km/s, no macroturbulence)."""
    q = json.loads((ROOT / "research" / "quiet_host.json").read_text())
    return Host("B", f"late-B star ({q['teff']:.0f} K)", q["teff"], q["logg"], q["mstar"], q["mmix_msun"],
                q["mmix_msun"], q["mmix_msun"], 1.0, 10.0, 0.0)


def host_grid() -> list[Host]:
    sun_cz = pinsonneault_mcz(5777)
    return [
        Host("M", "M dwarf (3500 K)", 3500, 5.0, 0.40, 0.35, 0.30, 0.40, 1.0, 2.0, 1.0),
        Host("K", "K dwarf (4500 K)", 4500, 4.5, 0.70, pinsonneault_mcz(4500),
             pinsonneault_mcz(4500) / 10**0.15, pinsonneault_mcz(4500) * 10**0.15, 1.0, 2.0, 2.0),
        Host("G", "Sun (5777 K)", 5777, 4.44, 1.00, sun_cz, sun_cz / 10**0.15, sun_cz * 10**0.15, 1.0, 1.9, 3.5,
             solar=True),
        Host("F", "F dwarf (6250 K)", 6250, 4.5, 1.25, pinsonneault_mcz(6250),
             pinsonneault_mcz(6250) / 10**0.15, pinsonneault_mcz(6250) * 10**0.15, 1.5, 8.0, 5.0),
        Host("Am", "Am star (8000 K)", 8000, 4.0, 1.8, *_a_star_mix(1.8), 2.0, 15.0, 3.0),
        Host("A0", "A0 star (9500 K)", 9500, 4.0, 2.3, *_a_star_mix(2.3), 2.0, 15.0, 3.0),
        _quiet_b(),
    ]


def isotope_mass(log_eps: float, A: float, mmix_msun: float) -> float:
    """Mass (g) of nuclide of mass A (u) that raises the mixed layer to abundance log eps."""
    n_ratio = 10.0 ** (log_eps - 12.0)
    return n_ratio * (A / M_H) * X_HYDROGEN * mmix_msun * MSUN


# Nuclides: half-life (yr) from the IAEA Live Chart of Nuclides (ENSDF), cumulative thermal-neutron
# fission yield of 235U per fission (IAEA, accessed 2026-10-28; research/iaea/).
NUCLIDES = {
    "99Tc": dict(z=43, A=98.906, t_half=2.111e5, fy=0.061316),
    "98Tc": dict(z=43, A=97.907, t_half=4.2e6, fy=None),
    "97Tc": dict(z=43, A=96.906, t_half=4.21e6, fy=None),
    "147Pm": dict(z=61, A=146.915, t_half=2.6234, fy=0.02232),
    "145Pm": dict(z=61, A=144.913, t_half=17.7, fy=None),
    "226Ra": dict(z=88, A=226.025, t_half=1600.0, fy=None),
}
E_FISSION = 200e6 * EV  # recoverable energy per fission (erg), round value


def fission_energy(mass_g: float, nuclide: str) -> float:
    """Thermal energy (erg) released by the fissions that make mass_g of a fission-product nuclide."""
    nd = NUCLIDES[nuclide]
    atoms_needed = mass_g / (nd["A"] * AMU)
    return atoms_needed / nd["fy"] * E_FISSION


# Production routes and their energy cost per atom produced.
#  fission:  harvest the nuclide from fission products; E_FISSION / (cumulative yield).
#  capture:  one neutron capture on an isotopically pure stable target, the neutron taken from
#            the spare neutrons of 235U fission: E_FISSION / (nu_bar - 1), with nu_bar = 2.46
#            (thermal 235U; Diven et al. 1956). Ideal bound: every spare neutron is captured.
#  U chain:  extract 226Ra from uranium in secular equilibrium (no energy cost assigned here).
NU_BAR = 2.46
CAPTURE_TARGET = {"99Tc": "98Mo", "97Tc": "96Ru", "145Pm": "144Sm"}
ROUTES = {
    "99Tc fission": ("99Tc", "fission"),
    "99Tc capture": ("99Tc", "capture"),
    "97Tc capture": ("97Tc", "capture"),
    "147Pm fission": ("147Pm", "fission"),
    "145Pm capture": ("145Pm", "capture"),
}
U238_FRACTION = 0.992742  # natural abundance of 238U (IAEA)
T_HALF_U238 = 4.468e9  # yr (IAEA)


def energy_per_atom(route: str) -> float:
    """Energy (erg) spent per atom of product along a production route."""
    nuc, kind = ROUTES[route]
    if kind == "fission":
        return E_FISSION / NUCLIDES[nuc]["fy"]
    return E_FISSION / (NU_BAR - 1.0)


def production_energy(mass_g: float, route: str) -> float:
    """Energy (erg) to make mass_g of the route's nuclide."""
    nuc, _ = ROUTES[route]
    return mass_g / (NUCLIDES[nuc]["A"] * AMU) * energy_per_atom(route)


def uranium_per_radium() -> float:
    """Mass of natural uranium (g) holding 1 g of 226Ra in secular equilibrium with 238U."""
    return (T_HALF_U238 / NUCLIDES["226Ra"]["t_half"]) * (238.0508 / NUCLIDES["226Ra"]["A"]) / U238_FRACTION


def upkeep_power(mass_g: float, route: str) -> float:
    """Power (erg s^-1) that replaces the decay of an inventory mass_g made along a route."""
    nuc, _ = ROUTES[route]
    return production_energy(resupply_rate(mass_g, nuc), route) / YR


def resupply_rate(mass_g: float, nuclide: str) -> float:
    """Mass per year (g yr^-1) that holds a steady inventory against radioactive decay."""
    return mass_g * np.log(2) / NUCLIDES[nuclide]["t_half"]


def to_earth(mass_g: float) -> float:
    return mass_g / MEARTH


__all__ = ["Host", "host_grid", "isotope_mass", "NUCLIDES", "fission_energy", "resupply_rate", "YR"]
