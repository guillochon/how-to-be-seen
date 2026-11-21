"""Radiative accelerations on trace elements in A and late-B atmospheres (LTE).

The acceleration on ion i of an element with atomic mass m is

    g_rad,i(depth) = 1 / (m c) * sum_lines (pi e^2 / m_e c) gf exp(-E_l / kT) / U_i
                     * int phi_l(nu) F_nu(nu, depth) dnu,

with F_nu the net flux at that depth. Each line is solved in a frequency window with the full
opacity there: continuum, the line itself, every other line of the element at the same abundance,
and the strong lines of other elements (Kurucz gfall, solar-scaled abundances of the ATLAS9 model).
In LTE the flux at each frequency depends only on the opacity at that frequency and on T(tau), so
the window transfer includes the line's own saturation and the blanketing by its neighbours.

Ionization balance and partition functions are Kurucz's (atoms/ZZII/partfnZZII.dat, potential
lowering column -500 cm^-1); the element's acceleration is the mean over ionization stages
weighted by fraction and by an approximate diffusion weight 1/Z^2 for ions.

Not included: metal bound-free opacity (see continuum_check), momentum given to the electron
when the absorbing ion is photoionized, and departures from LTE.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.special import expn, wofz

from . import KURUCZ, atlas, atoms
from .const import AMU, EV, LINE_CROSS, C, H, K
from .synth import Synth, planck_lambda

HC_K = H * C / K
KEV = K / EV
SAHA = 2.4147e15  # (2 pi m_e k / h^2)^1.5 in cm^-3 K^-1.5
MASS = {6: 12.011, 12: 24.305, 13: 26.982, 14: 28.086, 20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942,
        24: 51.996, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 43: 98.906}
# Neighbour species: (Z, ion) whose gfall lists are in research/kurucz/gfall.
NEIGHBOURS = [(6, 1), (12, 0), (12, 1), (13, 1), (14, 1), (14, 2), (20, 1), (21, 1), (22, 1), (23, 1),
              (24, 0), (24, 1), (24, 2), (25, 0), (25, 1), (25, 2), (26, 0), (26, 1), (26, 2), (27, 0),
              (27, 1), (28, 1), (28, 2)]
DIFF_W = {0: 1.0, 1: 1.0, 2: 0.25}


@cache
def partfn(z: int, ion: int):
    """(log10 T grid, log10 U, ionization energy in eV) from Kurucz's partfn file."""
    path = KURUCZ / ("tc" if z == 43 else "partfn") / f"partfn{z:02d}{ion:02d}.dat"
    if not path.exists():
        return None
    txt = [ln for ln in path.read_text().splitlines() if ln.strip()]
    ip = float(txt[0].split("potential")[-1].split()[0]) / 8065.544
    rows = [ln.split() for ln in txt[3:] if len(ln.split()) >= 4]
    lt = np.array([float(r[1]) for r in rows])
    u = np.array([float(r[3]) for r in rows])
    return lt, np.log10(np.maximum(u, 1e-30)), ip


@cache
def nist_ground() -> dict:
    """(Z, ion) -> (statistical weight of the ground level, ionization energy in eV) from the NIST ASD
    (scripts/fetch_gfall.py); for elements that Kurucz gives no partition function."""
    path = KURUCZ / "nist_ie.csv"
    out = {}
    if path.exists():
        with path.open() as fh:
            for r in csv.DictReader(fh):
                j = r["ground"].split("<")[-1].rstrip(">") if "<" in r["ground"] else r["ground"][-1]
                num, _, den = j.partition("/")
                out[int(r["z"]), int(r["ion"])] = (2 * float(num) / float(den or 1) + 1, float(r["ie_ev"]))
    return out


def U(z: int, ion: int, T: np.ndarray) -> np.ndarray:
    pf = partfn(z, ion)
    if pf is None:
        if z not in MASS and (z, ion) in nist_ground():  # ground level only
            return np.full_like(T, nist_ground()[z, ion][0])
        return np.full_like(T, 9.0)  # C I: ground term only (3P, g = 9), irrelevant for the ion fractions here
    return 10 ** np.interp(np.log10(T), pf[0], pf[1])


def ion_fractions(z: int, T: np.ndarray, ne: np.ndarray) -> np.ndarray:
    """Fractions in stages I, II, III, shape [3, ndepth]."""
    ips = [partfn(z, i)[2] if partfn(z, i)
           else nist_ground()[z, i][1] if z not in MASS and (z, i) in nist_ground()
           else atoms.IONIZATION.get(z, (11.26, 24.38))[i] for i in range(2)]
    us = [U(z, i, T) for i in range(3)]
    r = [np.ones_like(T)]
    for i in range(2):
        r.append(r[-1] * 2 * us[i + 1] / us[i] * SAHA * T**1.5 * np.exp(-ips[i] / (KEV * T)) / ne)
    r = np.array(r)
    return r / r.sum(axis=0)


@dataclass
class LineSet:
    z: int
    ion: np.ndarray
    wl: np.ndarray  # Angstrom
    loggf: np.ndarray
    elow: np.ndarray  # cm^-1
    gamma: np.ndarray  # log10 radiative, Stark, van der Waals (Kurucz conventions), shape [n, 3]


def _gfall(z: int, ion: int, wmin=880.0, wmax=20500.0, min_loggf=-4.0) -> LineSet:
    rows = []
    for wl, loggf, _c, e1, _j1, e2, _j2, gr, gs, gw in atoms._kurucz_rows(KURUCZ / "gfall" / f"gf{z:02d}{ion:02d}.all"):
        if wmin <= 10 * wl <= wmax and loggf > min_loggf:
            rows.append((10 * wl, loggf, min(abs(e1), abs(e2)), gr, gs, gw))
    a = np.array(rows) if rows else np.zeros((0, 6))
    return LineSet(z, np.full(len(a), ion), a[:, 0], a[:, 1], a[:, 2], a[:, 3:6])


def tc_lines() -> LineSet:
    """Tc I and III from Kurucz's computed lists (scripts/fetch_tc_lines.py), Tc II from gf4301.pos."""
    parts = []
    for ion in (0, 2):
        with (KURUCZ / "tc" / f"tc{ion}_lines.csv").open() as fh:
            r = [(float(d["wl_A"]), float(d["loggf"]), float(d["elow_cm"])) for d in csv.DictReader(fh)]
        a = np.array(r)
        parts.append((np.full(len(a), ion), a[:, 0], a[:, 1], a[:, 2]))
    t2 = atoms.kurucz_lines("gf4301.pos", 43, 1, 880.0, 20500.0)
    parts.append((np.full(len(t2), 1), np.array([ln.wl for ln in t2]), np.array([ln.loggf for ln in t2]),
                  np.array([ln.elow for ln in t2])))
    ion, wl, gf, el = (np.concatenate(x) for x in zip(*parts, strict=True))
    return LineSet(43, ion, wl, gf, el, np.zeros((len(wl), 3)))


class Atmosphere:
    """An ATLAS9 model with number densities, continuum opacities and line populations."""

    def __init__(self, teff: float, logg: float, vturb: float = 1.0, ndepth: int = 72):
        self.s = Synth(atlas.model(teff, logg), vturb=vturb, nrefine=ndepth)
        self.T, self.ne, self.rho, self.rhox = self.s.m.T, self.s.m.ne, self.s.st.rho, self.s.m.rhox
        self.vturb = vturb * 1e5
        self.nH = self.s.st.nH
        self.logg = logg
        self.tau5 = self.s.tau_cont_500()
        self._frac = {}
        self._kc = {}

    def frac(self, z):
        if z not in self._frac:
            self._frac[z] = ion_fractions(z, self.T, self.ne)
        return self._frac[z]

    def kcont(self, wl):
        key = round(wl, 0)
        if key not in self._kc:
            self._kc[key] = self.s.st.kappa_cont(wl)
        return self._kc[key]

    def line_strength(self, ls: LineSet, log_eps: float) -> np.ndarray:
        """Integrated line opacity per volume (cm^-1 Hz), shape [nline, ndepth]."""
        nX = self.nH * 10.0 ** (log_eps - 12.0)
        out = np.empty((len(ls.wl), len(self.T)))
        for ion in np.unique(ls.ion):
            sel = ls.ion == ion
            pop = nX * self.frac(ls.z)[ion] / U(ls.z, int(ion), self.T)
            boltz = np.exp(-HC_K * ls.elow[sel, None] / self.T[None, :])
            stim = 1.0 - np.exp(-HC_K / (ls.wl[sel, None] * 1e-8) / self.T[None, :])
            out[sel] = LINE_CROSS * 10.0 ** ls.loggf[sel, None] * pop[None, :] * boltz * stim
        return out

    def doppler(self, wl, mass_u):
        return C / (wl * 1e-8) / C * np.sqrt(2 * K * self.T / (mass_u * AMU) + self.vturb**2)

    def damping(self, ls: LineSet, idx, dnu):
        g = ls.gamma[idx]
        grad = np.where(g[:, 0] != 0, 10.0 ** g[:, 0], 2.223e15 / ls.wl[idx] ** 2)
        g4 = 10.0 ** np.where(g[:, 1] != 0, g[:, 1], -6.0)
        g6 = 10.0 ** np.where(g[:, 2] != 0, g[:, 2], -7.7)
        gam = grad[:, None] + g4[:, None] * self.ne * (self.T / 1e4) ** (1 / 6) + g6[:, None] * self.s.st.nHI * (self.T / 1e4) ** 0.3
        return gam / (4 * np.pi * dnu)


def net_flux(tau: np.ndarray, S: np.ndarray, dtau_min: float = 1e-30) -> np.ndarray:
    """Net flux F(tau_d) = 2 pi [int_tau^inf S E2(t - tau) dt - int_0^tau S E2(tau - t) dt] for each
    column; tau and S have shape [ndepth, nfreq]. S is extended below the model by its gradient.
    dtau_min bounds the optical-depth step used for that gradient: where the model is transparent
    at its base (cool stars just longward of the Lyman edge, with no metal opacity) the gradient
    would otherwise extrapolate S to absurd values."""
    nd, nf = tau.shape
    ext = tau[-1] + np.geomspace(1e-3, 40.0, 30)[:, None] * np.ones((1, nf))
    dS = (S[-1] - S[-2]) / np.maximum(tau[-1] - tau[-2], max(dtau_min, 1e-30))
    tt = np.vstack([np.zeros((1, nf)), tau, ext])
    SS = np.vstack([S[:1], S, S[-1] + dS * (ext - tau[-1])])
    out = np.empty((nd, nf))
    for d in range(nd):
        x = tt - tau[d]
        w = np.where(x >= 0, expn(2, np.abs(x)), -expn(2, np.abs(x)))
        out[d] = 2 * np.pi * np.trapezoid(SS * w, tt, axis=0)
    # Deep in the medium the two half-integrals cancel and the quadrature loses all precision;
    # there the diffusion limit F = (4 pi / 3) dS/dtau is exact to O(1/tau^2).
    deep = tau > 10.0
    out[deep] = (4 * np.pi / 3) * (np.gradient(S, axis=0)[deep] / np.gradient(tau, axis=0)[deep])
    return out


def continuum_flux(atm: Atmosphere, wls: np.ndarray) -> np.ndarray:
    """F_lambda at every depth for the continuum alone, shape [ndepth, nwl]."""
    tau = np.empty((len(atm.T), len(wls)))
    S = np.empty_like(tau)
    for j, wl in enumerate(wls):
        x = atm.kcont(wl) / atm.rho
        tau[:, j] = np.concatenate([[x[0] * atm.rhox[0]], x[0] * atm.rhox[0] + np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(atm.rhox))])
        S[:, j] = planck_lambda(wl, atm.T)
    return net_flux(tau, S, atm.tau5[-1] - atm.tau5[-2])


def castelli_flux(teff: float, logg: float):
    """(wavelength A, F_nu blanketed, F_nu continuum) at the surface from Castelli's fp00k2odfnew.pck."""
    txt = (KURUCZ / "fp00k2odfnew.pck").read_text().splitlines()
    i0 = next(k for k, ln in enumerate(txt) if ln.startswith("TEFF"))
    wave = np.array([float(ln[c:c + 10]) for ln in txt[22:i0] for c in range(0, len(ln.rstrip()), 10)])
    i = next(k for k, ln in enumerate(txt) if ln.split()[:4] == ["TEFF", f"{teff:.0f}.", "GRAVITY", f"{logg:.5f}"])
    vals = []
    k = i + 1
    while len(vals) < 2 * len(wave):
        ln = txt[k]
        vals += [float(ln[c:c + 10]) for c in range(0, len(ln.rstrip()), 10)]
        k += 1
    h = np.array(vals)
    return 10 * wave, 4 * np.pi * h[:len(wave)], 4 * np.pi * h[len(wave):]


def neighbour_lines(atm: Atmosphere, exclude_z: int, ratio_min: float = 1e-3) -> tuple[LineSet, np.ndarray, np.ndarray]:
    """Lines of other elements whose centre opacity exceeds ratio_min of the continuum somewhere above
    tau5000 = 3, with their integrated opacity [n, ndepth] and atomic masses."""
    keep_wl, keep_k, keep_m, keep_g, keep_z = [], [], [], [], []
    deep = atm.tau5 < 3.0
    for z, ion in NEIGHBOURS:
        if z == exclude_z:
            continue
        ls = _gfall(z, ion)
        if not len(ls.wl):
            continue
        le = atm.s.m.log_abund[z] + 12.0
        k = atm.line_strength(ls, le)
        dnu = C / (ls.wl[:, None] * 1e-8) / C * np.sqrt(2 * K * atm.T / (MASS[z] * AMU) + atm.vturb**2)[None, :]
        kc = np.array([atm.kcont(w) for w in np.round(ls.wl, -1)])
        ratio = (k / (np.sqrt(np.pi) * dnu) / kc)[:, deep].max(axis=1)
        sel = ratio > ratio_min
        keep_wl.append(ls.wl[sel]); keep_k.append(k[sel]); keep_g.append(ls.gamma[sel])  # noqa: E702
        keep_m.append(np.full(sel.sum(), MASS[z])); keep_z.append(np.full(sel.sum(), z))  # noqa: E702
    wl = np.concatenate(keep_wl)
    o = np.argsort(wl)
    ls = LineSet(0, np.concatenate(keep_z)[o], wl[o], np.zeros(len(wl)), np.zeros(len(wl)), np.concatenate(keep_g)[o])
    return ls, np.concatenate(keep_k)[o], np.concatenate(keep_m)[o]


def _profile(atm, wl0, wl_c, k, mass, gam, x_nu):
    """Opacity [ndepth, nfreq] of lines (centres wl_c, integrated opacity k [n, nd]) at frequencies x_nu."""
    if len(wl_c) == 0:
        return np.zeros((len(atm.T), len(x_nu)))
    nu_c = C / (wl_c * 1e-8)
    dnu = nu_c[:, None] / C * np.sqrt(2 * K * atm.T[None, :] / (mass[:, None] * AMU) + atm.vturb**2)
    g = gam
    grad = np.where(g[:, 0] != 0, 10.0 ** g[:, 0], 2.223e15 / wl_c**2)
    g4 = 10.0 ** np.where(g[:, 1] != 0, g[:, 1], -6.0)
    g6 = 10.0 ** np.where(g[:, 2] != 0, g[:, 2], -7.7)
    gamma = grad[:, None] + g4[:, None] * atm.ne * (atm.T / 1e4) ** (1 / 6) + g6[:, None] * atm.s.st.nHI * (atm.T / 1e4) ** 0.3
    a = gamma / (4 * np.pi * dnu)
    v = (x_nu[None, None, :] - nu_c[:, None, None]) / dnu[:, :, None]
    phi = wofz(v + 1j * a[:, :, None]).real / (np.sqrt(np.pi) * dnu[:, :, None])
    return np.einsum("nd,ndf->df", k, phi)


def grad_element(atm: Atmosphere, el: LineSet, mass: float, log_eps: float, nb, flux_corr=None,
                 keep_frac: float = 0.99, max_lines: int = 2500, nfreq: int = 33):
    """Radiative acceleration on each ionization stage of an element, shape [3, ndepth], with every
    line solved in its own window (own saturation plus neighbours). Returns (g_ion, nlines used)."""
    nb_ls, nb_k, nb_m = nb
    T, nd = atm.T, len(atm.T)
    frac = atm.frac(el.z)
    k_el = atm.line_strength(el, log_eps)
    nX = atm.nH * 10.0 ** (log_eps - 12.0)
    # unsaturated contribution with the continuum flux, to choose the lines that matter
    grid = np.geomspace(880.0, 20500.0, 160)
    Fc = continuum_flux(atm, grid)
    lw = np.log(grid)
    Fl = np.stack([np.interp(np.log(el.wl), lw, Fc[d]) for d in range(nd)], axis=1)  # [n, nd]
    contrib = k_el * Fl * (el.wl[:, None] * 1e-8) ** 2 / C
    chosen = set()
    zone = np.nonzero((atm.tau5 > 1e-4) & (atm.tau5 < 5.0))[0]
    for ion in range(3):
        sel = np.nonzero(el.ion == ion)[0]
        if not len(sel):
            continue
        for d in zone[::3]:
            c = contrib[sel, d]
            o = np.argsort(c)[::-1]
            cum = np.cumsum(c[o]) / max(c.sum(), 1e-300)
            n = min(int(np.searchsorted(cum, keep_frac)) + 1, max_lines)
            chosen.update(sel[o[:n]].tolist())
    chosen = np.array(sorted(chosen))
    g_ion = np.zeros((3, nd))
    xg = np.concatenate([-np.geomspace(8, 0.05, nfreq // 2), [0.0], np.geomspace(0.05, 8, nfreq // 2)])
    el_m = np.full(len(el.wl), mass)
    for i in chosen:
        wl0 = el.wl[i]
        nu0 = C / (wl0 * 1e-8)
        dnu0 = nu0 / C * np.sqrt(2 * K * T / (mass * AMU) + atm.vturb**2)
        x_nu = nu0 + xg * dnu0.mean()
        wl_x = C / x_nu * 1e8
        w = 1.0  # Angstrom: lines whose centres lie within the window +- w contribute
        a, b = np.searchsorted(nb_ls.wl, [wl0 - w, wl0 + w])
        kap = _profile(atm, wl0, nb_ls.wl[a:b], nb_k[a:b], nb_m[a:b], nb_ls.gamma[a:b], x_nu)
        near = np.nonzero(np.abs(el.wl - wl0) < w)[0]
        kap += _profile(atm, wl0, el.wl[near], k_el[near], el_m[near], el.gamma[near], x_nu)
        kap += atm.kcont(wl0)[:, None]
        x = kap / atm.rho[:, None]
        tau = np.vstack([x[:1] * atm.rhox[0], x[:1] * atm.rhox[0] + np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(atm.rhox)[:, None], axis=0)])
        S = np.stack([planck_lambda(wv, T) for wv in wl_x], axis=1)
        F = net_flux(tau, S, atm.tau5[-1] - atm.tau5[-2]) * (wl_x[None, :] * 1e-8) ** 2 / C  # F_nu
        if flux_corr is not None:
            F *= flux_corr(wl0)
        phi_self = _profile(atm, wl0, el.wl[i:i + 1], k_el[i:i + 1], el_m[i:i + 1], el.gamma[i:i + 1], x_nu)
        mom = np.trapezoid(phi_self * F, x_nu, axis=1) / C  # momentum per volume per time
        ion = int(el.ion[i])
        n_ion = nX * frac[ion]
        g_ion[ion] += mom / np.maximum(n_ion * mass * AMU, 1e-300)
    return g_ion, len(chosen)


def element_grad(atm: Atmosphere, g_ion: np.ndarray, z: int):
    """(fraction-weighted, diffusion-weighted) element accelerations."""
    f = atm.frac(z)
    plain = sum(f[i] * g_ion[i] for i in range(3))
    wts = sum(f[i] * DIFF_W[i] for i in range(3))
    return plain, sum(f[i] * DIFF_W[i] * g_ion[i] for i in range(3)) / wts


def flux_correction(atm: Atmosphere, teff: float, logg: float):
    """Castelli continuum / our continuum at the surface, applied below 2000 A where our continuum
    lacks metal bound-free opacity (1 above)."""
    w, _, fc = castelli_flux(teff, logg)
    grid = np.array([900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000.0])
    ours = continuum_flux(atm, grid)[0] * (grid * 1e-8) ** 2 / C
    ratio = np.clip(np.interp(grid, w, fc) / ours, 0.0, 1.0)
    return lambda wl: float(np.interp(wl, grid, ratio)) if wl < 2000 else 1.0
