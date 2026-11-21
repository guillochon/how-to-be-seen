"""LTE formal solution for the flux in and near one spectral line.

The emergent flux is F = 2 pi int B(T(tau)) E_2(tau) dtau along each wavelength, with the
total opacity (continuum + line) integrated on the model's column-mass scale. The line
profile is a Voigt function with radiative, quadratic Stark and van der Waals damping.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expn, wofz

from . import atoms
from .atlas import Model
from .const import AMU, LINE_CROSS, C, H, K
from .opacity import State


def refine(m: Model, n: int = 240) -> Model:
    """Interpolate a model onto a finer column-mass grid (log-log), keeping its abundances."""
    lx = np.log10(m.rhox)
    x = np.linspace(lx[0], lx[-1], n)
    f = lambda y: 10.0 ** np.interp(x, lx, np.log10(y))  # noqa: E731
    return Model(m.teff, m.logg, m.vturb, 10.0**x, f(m.T), f(m.P), f(m.ne), m.log_abund, m.he_over_h)


def planck_lambda(wl_A: float, T: np.ndarray) -> np.ndarray:
    lam = wl_A * 1e-8
    return 2 * H * C**2 / lam**5 / np.expm1(H * C / (lam * K * T))


def _flux(tau: np.ndarray, S: np.ndarray) -> np.ndarray:
    """2 pi int S E2 dtau for each column (tau, S have shape [ndepth, nwl])."""
    tau = np.vstack([np.zeros((1, tau.shape[1])), tau])
    S = np.vstack([S[:1], S])
    integrand = S * expn(2, tau)
    return 2 * np.pi * np.trapezoid(integrand, tau, axis=0)


class Synth:
    """Line formation in one model atmosphere."""

    def __init__(self, m: Model, vturb: float | None = None, nrefine: int = 240, metals: bool = True):
        self.m = refine(m, nrefine)
        self.vturb = (m.vturb if vturb is None else vturb) * 1e5  # cm s^-1
        self.st = State(self.m, metals)
        self._frac = {}

    def fraction(self, z: int, ion: int) -> np.ndarray:
        if z not in self._frac:
            self._frac[z] = self.st.ion_fractions(z, atoms.IONIZATION[z], atoms.partition)
        return self._frac[z][ion]

    def line_opacity_center_factor(self, line: atoms.Line, log_eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (integrated line opacity per volume [cm^-1 Hz], Doppler width [Hz], damping a)."""
        T, ne = self.m.T, self.m.ne
        nX = self.st.nH * 10.0 ** (log_eps - 12.0)
        U = atoms.partition(line.z, line.ion, T)
        nlow_over_g = nX * self.fraction(line.z, line.ion) * np.exp(-H * C * line.elow / (K * T)) / U
        nu0 = C / (line.wl * 1e-8)
        stim = 1.0 - np.exp(-H * nu0 / (K * T))
        kint = LINE_CROSS * 10.0**line.loggf * nlow_over_g * stim
        mass = atoms.MASS[line.z] * AMU
        dnu = nu0 / C * np.sqrt(2 * K * T / mass + self.vturb**2)
        grad = 10.0 ** line.gamma_rad if line.gamma_rad else 2.223e15 / line.wl**2
        g4 = 10.0 ** (line.gamma_stark if line.gamma_stark else -6.0)
        g6 = 10.0 ** (line.gamma_waals if line.gamma_waals else -7.7)
        gamma = grad + g4 * ne * (T / 1e4) ** (1 / 6) + g6 * self.st.nHI * (T / 1e4) ** 0.3
        a = gamma / (4 * np.pi * dnu)
        return kint, dnu, a

    def profile(self, line: atoms.Line, log_eps: float, half_width: float | None = None, npts: int = 241):
        """Normalized flux across the line: returns (wavelength [A], F/F_cont, F_cont)."""
        hw = half_width or 1.5e-4 * line.wl  # +/- 45 km/s
        wl = line.wl + np.linspace(-hw, hw, npts)
        kc = self.st.kappa_cont(line.wl)  # continuum varies negligibly across the line
        kint, dnu, a = self.line_opacity_center_factor(line, log_eps)
        nu = C / (wl * 1e-8)
        nu0 = C / (line.wl * 1e-8)
        v = (nu[None, :] - nu0) / dnu[:, None]
        phi = wofz(v + 1j * a[:, None]).real / (np.sqrt(np.pi) * dnu[:, None])
        k_tot = kc[:, None] + kint[:, None] * phi
        rho = self.st.rho[:, None]
        dtau = 0.5 * (k_tot[1:] / rho[1:] + k_tot[:-1] / rho[:-1]) * np.diff(self.m.rhox)[:, None]
        tau0 = k_tot[:1] / rho[:1] * self.m.rhox[0]
        tau = np.vstack([tau0, tau0 + np.cumsum(dtau, axis=0)])
        S = planck_lambda(line.wl, self.m.T)[:, None] * np.ones_like(tau)
        F = _flux(tau, S)
        tauc = np.concatenate([[kc[0] / self.st.rho[0] * self.m.rhox[0]],
                               kc[0] / self.st.rho[0] * self.m.rhox[0]
                               + np.cumsum(0.5 * (kc[1:] / self.st.rho[1:] + kc[:-1] / self.st.rho[:-1]) * np.diff(self.m.rhox))])
        Fc = _flux(tauc[:, None], planck_lambda(line.wl, self.m.T)[:, None])[0]
        return wl, F / Fc, Fc

    def family(self, line: atoms.Line, wl: np.ndarray) -> LineFamily:
        """Profiles of one line on the wavelengths wl at any abundance (see LineFamily)."""
        return LineFamily(self, line, wl)

    def ew(self, line: atoms.Line, log_eps: float) -> float:
        """Equivalent width (milli-Angstrom)."""
        wl, f, _ = self.profile(line, log_eps)
        return 1e3 * np.trapezoid(1.0 - f, wl)

    def log_eps_for_ew(self, line: atoms.Line, ew_mA: float, lo: float = -8.0, hi: float = 8.0) -> float:
        """Abundance log eps = log(N_X/N_H) + 12 that gives the requested equivalent width."""
        g = lambda le: np.log10(max(self.ew(line, le), 1e-12)) - np.log10(ew_mA)  # noqa: E731
        return brentq(g, lo, hi, xtol=1e-3)

    def tau_cont_500(self) -> np.ndarray:
        kc = self.st.kappa_cont(5000.0)
        x = kc / self.st.rho
        return np.concatenate([[x[0] * self.m.rhox[0]], x[0] * self.m.rhox[0] + np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(self.m.rhox))])


class LineFamily:
    """Normalized flux of one trace line on fixed wavelengths, for any abundance.

    The line opacity of a trace element is proportional to its abundance, and nothing else in the
    atmosphere depends on it. The continuum opacity, Voigt profile and source function are therefore
    computed once, and each abundance costs only the optical-depth integral and the flux integral.
    Synth.profile gives the same result for one abundance.
    """

    def __init__(self, s: Synth, line: atoms.Line, wl: np.ndarray):
        self.wl = np.asarray(wl, float)
        kc = s.st.kappa_cont(line.wl)
        kint, dnu, a = s.line_opacity_center_factor(line, 12.0)  # abundance scale: log eps = 12
        nu = C / (self.wl * 1e-8)
        nu0 = C / (line.wl * 1e-8)
        v = (nu[None, :] - nu0) / dnu[:, None]
        phi = wofz(v + 1j * a[:, None]).real / (np.sqrt(np.pi) * dnu[:, None])
        rho = s.st.rho[:, None]
        self._kc = kc[:, None] / rho
        self._kl = kint[:, None] * phi / rho
        self._drhox = np.diff(s.m.rhox)[:, None]
        self._rhox0 = s.m.rhox[0]
        self._S = planck_lambda(line.wl, s.m.T)[:, None] * np.ones((1, len(self.wl)))
        x = kc / s.st.rho
        tauc = np.concatenate([[x[0] * s.m.rhox[0]], x[0] * s.m.rhox[0] + np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(s.m.rhox))])
        self._Fc = _flux(tauc[:, None], planck_lambda(line.wl, s.m.T)[:, None])[0]

    def __call__(self, log_eps: float) -> np.ndarray:
        x = self._kc + 10.0 ** (log_eps - 12.0) * self._kl
        tau0 = x[:1] * self._rhox0
        tau = np.vstack([tau0, tau0 + np.cumsum(0.5 * (x[1:] + x[:-1]) * self._drhox, axis=0)])
        return _flux(tau, self._S) / self._Fc
