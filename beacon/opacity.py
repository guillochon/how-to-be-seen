"""Continuous opacity and the number densities it needs.

Included: H- bound-free and free-free (John 1988 fits), hydrogenic H I bound-free (levels
n = 1-8, Gaunt factor 1) and free-free, Rayleigh scattering by H I, Thomson scattering, and the
bound-free absorption of C, N, O, Mg, Al, Si, Ca and Fe that ATLAS includes (beacon/metals.py).
State(m, metals=False) leaves the metals out, as the first version of the paper did.
"""

from __future__ import annotations

import numpy as np

from .atlas import Model
from .const import EV, KEV, SAHA, SIGMA_T, C, H, K
from .metals import kappa_metals

CHI_HMINUS = 0.754195  # H- binding energy (eV)
CHI_H = 13.598434599702

# John (1988, A&A 193, 189): H- bound-free cross-section coefficients.
_BF_C = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
_LAMBDA0 = 1.6419  # micron

# John (1988): H- free-free, lambda > 0.3645 micron. Rows n = 1..6; columns A..F.
_FF_LONG = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2483.346, 285.827, -2054.291, 2827.776, -1341.537, 208.952],
    [-3449.889, -1158.382, 8746.523, -11485.632, 5303.609, -812.939],
    [2200.040, 2427.719, -13651.105, 16755.524, -7510.494, 1132.738],
    [-696.271, -1841.400, 8624.970, -10051.530, 4400.067, -655.020],
    [88.283, 444.517, -1863.864, 2095.288, -901.788, 132.985],
])
# John (1988): H- free-free, 0.1823 < lambda < 0.3645 micron.
_FF_SHORT = np.array([
    [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
    [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
    [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
    [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])


def hminus_bf_cross(wl_um: float) -> float:
    """H- photodetachment cross-section (cm^2) at wavelength wl_um (micron)."""
    if wl_um >= _LAMBDA0:
        return 0.0
    x = 1.0 / wl_um - 1.0 / _LAMBDA0
    f = sum(c * x ** (n / 2.0) for n, c in enumerate(_BF_C))
    return 1e-18 * wl_um**3 * x**1.5 * f


def hminus_ff_coeff(wl_um: float, T: np.ndarray) -> np.ndarray:
    """H- free-free absorption per neutral H atom per unit electron pressure (cm^4 dyn^-1)."""
    tab = _FF_LONG if wl_um > 0.3645 else _FF_SHORT
    th = 5040.0 / T
    lam = wl_um
    tot = 0.0
    for n in range(6):
        a, b, c, d, e, f = tab[n]
        tot = tot + th ** ((n + 2) / 2.0) * (lam**2 * a + b + c / lam + d / lam**2 + e / lam**3 + f / lam**4)
    return 1e-29 * tot


class State:
    """Number densities on the depth grid of a model atmosphere."""

    def __init__(self, m: Model, metals: bool = True):
        self.m = m
        self.metals = metals
        T, ne = m.T, m.ne
        ntot = m.P / (K * T)
        nnuc = ntot - ne
        metals = np.sum(10.0 ** m.log_abund[3:])
        self.nH = nnuc / (1.0 + m.he_over_h + metals)  # all hydrogen nuclei (cm^-3)
        # H I / H II from Saha with U(H I) = 2, U(H II) = 1
        r = 2.0 * (1.0 / 2.0) * SAHA * T**1.5 * np.exp(-CHI_H / (KEV * T)) / ne
        self.nHI = self.nH / (1.0 + r)
        self.nHII = self.nH - self.nHI
        # H-: n(H-) = n(H I) n_e / [4 (2 pi m k T / h^2)^1.5] exp(chi/kT), U(H-) = 1
        self.nHm = self.nHI * ne / (4.0 * SAHA * T**1.5) * np.exp(CHI_HMINUS / (KEV * T))
        self.rho = self.nH * 1.6735e-24 * (1.0 + 3.97 * m.he_over_h) + metals * self.nH * 30 * 1.6605e-24

    def ion_fractions(self, z: int, ip: tuple[float, ...], partition) -> np.ndarray:
        """Fractions in stages I, II, III (shape [3, ndepth]) from the Saha equation."""
        T, ne = self.m.T, self.m.ne
        U = [partition(z, i, T) for i in range(3)]
        ratios = [np.ones_like(T)]
        for i in range(2):
            r = 2.0 * U[i + 1] / U[i] * SAHA * T**1.5 * np.exp(-ip[i] / (KEV * T)) / ne
            ratios.append(ratios[-1] * r)
        ratios = np.array(ratios)
        return ratios / ratios.sum(axis=0)

    def kappa_cont(self, wl_A: float) -> np.ndarray:
        """Continuous absorption + scattering coefficient per unit volume (cm^-1) at wl_A (Angstrom)."""
        T, ne = self.m.T, self.m.ne
        nu = C / (wl_A * 1e-8)
        stim = 1.0 - np.exp(-H * nu / (K * T))
        wl_um = wl_A * 1e-4
        k = self.nHm * hminus_bf_cross(wl_um) * stim
        k = k + hminus_ff_coeff(wl_um, T) * self.nHI * ne * K * T
        # hydrogenic H I bound-free from n = 1..8 (Kramers cross-section, g_bf = 1)
        for n in range(1, 9):
            chi_n = CHI_H / n**2  # ionization energy from level n (eV)
            if H * nu / EV < chi_n:
                continue
            pop = self.nHI * (2 * n**2 / 2.0) * np.exp(-(CHI_H - chi_n) / (KEV * T))
            sig = 2.815e29 / (n**5 * nu**3)
            k = k + pop * sig * stim
        # H I free-free (g_ff = 1)
        k = k + 3.69e8 * ne * self.nHII / (np.sqrt(T) * nu**3) * stim
        # Rayleigh scattering by H I (Dalgarno, as used in ATLAS) and Thomson scattering
        w = max(wl_A, 1026.0)
        k = k + self.nHI * (5.799e-13 / w**4 + 1.422e-6 / w**6 + 2.784 / w**8)
        k = k + ne * SIGMA_T
        if self.metals:
            k = k + kappa_metals(self, wl_A)
        return k
