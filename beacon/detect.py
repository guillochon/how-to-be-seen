"""Equivalent-width detection limits and photon budgets for high-resolution spectrographs.

Noise on an equivalent width. Fit a line of known shape phi (unit area, Gaussian with
FWHM w) to a continuum-normalized spectrum sampled at pixel width dl with per-pixel noise
1/SNR. The least-squares (matched-filter) error is
    sigma_W = (1/SNR) * [dl / int phi^2 dlambda]^(1/2) = (1/SNR) * (w dl / 0.6643)^(1/2),
since int phi^2 = 2 sqrt(ln 2 / pi) / w = 0.6643 / w for a Gaussian. This is the form of
Cayrel (1988) without the continuum-placement term; CONT_FACTOR inflates it to allow for
the continuum (Cayrel's own coefficient is larger than the matched-filter one).
Several lines of one species combine as a matched filter: S/N_tot^2 = sum (W_i/sigma_i)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .const import C, H

GAUSS_INT = np.sqrt(2 * np.log(2) / np.pi)  # w * int phi^2 = 0.6643 for a Gaussian of FWHM w
CONT_FACTOR = 1.3  # continuum-placement inflation of sigma_W (assumed)
PIX_PER_RES = 3.0  # detector pixels per resolution element (assumed)


def sigma_ew(wl_A: float, R: float, snr: float, fwhm_kms: float) -> float:
    """1-sigma equivalent-width error (milli-Angstrom) for one line."""
    w = wl_A * fwhm_kms / (C / 1e5)
    dl = wl_A / (R * PIX_PER_RES)
    return 1e3 * CONT_FACTOR * np.sqrt(w * dl / GAUSS_INT) / snr


def observed_fwhm(wl_A: float, R: float, intrinsic_kms: float, vsini: float = 0.0) -> float:
    """Observed line FWHM (km s^-1): intrinsic, rotational (FWHM ~ 1.6 v sin i) and instrumental in quadrature."""
    inst = C / 1e5 / R
    return float(np.sqrt(intrinsic_kms**2 + (1.6 * vsini) ** 2 + inst**2))


@dataclass
class Instrument:
    name: str
    D: float  # telescope aperture (m)
    R: float  # resolving power
    eff: float  # total photon-to-electron efficiency (telescope + instrument + detector; assumed)
    wmin: float  # wavelength coverage (Angstrom)
    wmax: float
    space: bool = False


# Representative instruments. Apertures, resolving powers and coverages are the published ones;
# total efficiencies are round assumed values (see paper Sect. 3.4), not instrument specifications.
INSTRUMENTS = [
    Instrument("3.6 m, R = 115,000", 3.6, 115_000, 0.05, 3780, 6910),
    Instrument("8.2 m, R = 140,000", 8.2, 140_000, 0.08, 3800, 7880),
    Instrument("8.2 m UV-blue, R = 80,000", 8.2, 80_000, 0.08, 3050, 5000),
    Instrument("39 m, R = 100,000", 39.0, 100_000, 0.08, 3500, 18000),
    Instrument("2.4 m space UV, R = 114,000", 2.4, 114_000, 0.01, 1650, 3100, True),
    Instrument("6 m space UV-optical, R = 100,000", 6.0, 100_000, 0.05, 1000, 10000, True),
]

# Flux density of a V = 0 star at 5500 A (erg s^-1 cm^-2 A^-1), derived from the solar model in
# scripts/flux_zero_point.py.
FLAMBDA_V0 = 3.773e-9


def snr_per_pixel(inst: Instrument, V: float, t_s: float, wl_A: float, flux_ratio: float) -> float:
    """Photon-limited S/N per pixel at wl_A for a star of magnitude V observed for t_s seconds.

    flux_ratio is F_lambda(wl_A) / F_lambda(5500 A) for the star (from its model atmosphere).
    Read noise, sky and dark current are neglected: these are bright stars.
    """
    f = FLAMBDA_V0 * 10 ** (-0.4 * V) * flux_ratio
    photons = f / (H * C / (wl_A * 1e-8))  # photons s^-1 cm^-2 A^-1
    area = np.pi * (50.0 * inst.D) ** 2
    dl = wl_A / (inst.R * PIX_PER_RES)
    n = photons * area * inst.eff * dl * t_s
    return float(np.sqrt(n))


def time_for_snr(inst: Instrument, V: float, snr: float, wl_A: float, flux_ratio: float) -> float:
    """Exposure time (s) to reach a per-pixel S/N."""
    s1 = snr_per_pixel(inst, V, 1.0, wl_A, flux_ratio)
    return (snr / s1) ** 2
