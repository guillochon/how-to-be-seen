"""Equivalent-width curves for every beacon line in every host, and the abundance a search can detect.

The detection statistic for one species is the matched-filter significance summed over its
lines that fall inside an instrument's coverage:
    Z(log eps) = [ sum_i (W_i(log eps) / sigma_i)^2 ]^(1/2),
and the required abundance is where Z = Z_DETECT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ROOT, atlas, atoms, detect, hosts, synth
from .const import AMU, C, K

Z_DETECT = 5.0  # combined significance that counts as a detection
LOG_EPS_GRID = np.arange(-2.0, 9.01, 0.25)

SPECIES = {
    "Tc I": lambda: atoms.tc1_resonance(),
    "Tc II": lambda: atoms.tc2_lines(min_loggf=-1.0),
    "Pm II": lambda: atoms.pm2_lines(),
}


def absolute_v(teff: float) -> float:
    """M_V of a main-sequence star of this Teff (Pecaut & Mamajek 2013; Mamajek table v2022.04.16)."""
    teffs, mv = [], []
    for ln in (ROOT / "research" / "mamajek" / "EEM_dwarf_UBVIJHK_colors_Teff.txt").read_text().splitlines():
        tok = ln.split()
        if len(tok) > 8 and tok[0].endswith("V") and tok[0][0] in "BAFGKM" and tok[1].isdigit():
            try:
                teffs.append(float(tok[1]))
                mv.append(float(tok[7]))
            except ValueError:
                continue
    teffs, mv = np.array(teffs), np.array(mv)
    o = np.argsort(teffs)
    return float(np.interp(teff, teffs[o], mv[o]))


@dataclass
class Curves:
    host: hosts.Host
    species: str
    lines: list
    ew: np.ndarray  # [nline, nlogeps] equivalent widths (mA)
    fwhm_int: np.ndarray  # intrinsic line FWHM (km s^-1) per line
    flux_ratio: np.ndarray  # blanketed F_lambda(line) / F_lambda(5500 A) per line


def intrinsic_fwhm(host: hosts.Host, A: float) -> float:
    """Thermal + microturbulent Doppler FWHM combined with macroturbulence (km s^-1)."""
    b = np.sqrt(2 * K * host.teff / (A * AMU) + (host.vturb * 1e5) ** 2) / 1e5
    return float(np.sqrt((2 * np.sqrt(np.log(2)) * b) ** 2 + host.macro**2))


def blanketing(host: hosts.Host, wl_A) -> np.ndarray:
    """Ratio of line-blanketed to continuum flux, F_nu / F_nu,cont, at wl_A in the ATLAS9 flux grid of Castelli &
    Kurucz (2003), for the grid model nearest the host (the Sun uses 5750 K, log g = 4.5)."""
    from .levitation import castelli_flux

    teff = 5750.0 if host.solar else round(host.teff / 250.0) * 250.0
    logg = 4.5 if host.solar else round(host.logg * 2.0) / 2.0
    w, fb, fc = castelli_flux(teff, logg)
    return np.interp(wl_A, w, fb / fc)


def build_curves(host: hosts.Host) -> dict[str, Curves]:
    m = atlas.sun() if host.solar else atlas.model(host.teff, host.logg)
    s = synth.Synth(m, vturb=host.vturb)
    ref_line = atoms.Line(5500.0, -9.0, 0.0, 26, 0)
    _, _, fc5500 = s.profile(ref_line, -20.0, npts=3)
    b5500 = blanketing(host, 5500.0)
    out = {}
    for name, getter in SPECIES.items():
        lines = getter()
        ew = np.array([[s.ew(ln, le) for le in LOG_EPS_GRID] for ln in lines])
        # the photon budget uses the blanketed flux at the line, relative to the blanketed flux that V measures
        fr = np.array([s.profile(ln, -20.0, npts=3)[2] / fc5500 * blanketing(host, ln.wl) / b5500 for ln in lines])
        A = atoms.MASS[lines[0].z]
        fw = np.full(len(lines), intrinsic_fwhm(host, A))
        out[name] = Curves(host, name, lines, ew, fw, fr)
    return out


def ew_at(ew_row: np.ndarray, log_eps: float) -> float:
    """Equivalent width at log_eps; below the grid, weak lines scale linearly with abundance."""
    if log_eps < LOG_EPS_GRID[0]:
        return float(ew_row[0] * 10.0 ** (log_eps - LOG_EPS_GRID[0]))
    return float(10.0 ** np.interp(log_eps, LOG_EPS_GRID, np.log10(np.maximum(ew_row, 1e-30))))


def significance(c: Curves, inst: detect.Instrument, snr, log_eps: float) -> float:
    """Combined significance. snr is a fixed per-pixel S/N, or a tuple (V, t_s, cap) for a photon
    budget evaluated line by line and capped at a systematic floor `cap`."""
    z2 = 0.0
    for ln, ew_row, fw, fr in zip(c.lines, c.ew, c.fwhm_int, c.flux_ratio, strict=True):
        if not inst.wmin <= ln.wl <= inst.wmax:
            continue
        if isinstance(snr, tuple):
            V, t_s, cap = snr
            sn = min(detect.snr_per_pixel(inst, V, t_s, ln.wl, fr), cap)
        else:
            sn = snr
        w = ew_at(ew_row, log_eps)
        sig = detect.sigma_ew(ln.wl, inst.R, sn, detect.observed_fwhm(ln.wl, inst.R, fw, c.host.vsini))
        z2 += (w / sig) ** 2
    return float(np.sqrt(z2))


def required_log_eps(c: Curves, inst: detect.Instrument, snr) -> float:
    """Abundance at which the combined significance reaches Z_DETECT (nan if no line is covered)."""
    grid = np.concatenate([np.arange(-5.0, LOG_EPS_GRID[0], 0.25), LOG_EPS_GRID])
    zs = np.array([significance(c, inst, snr, le) for le in grid])
    if not np.any(zs > 0):
        return np.nan
    # Saturated lines overflow the synthesis window far above any threshold (Synth.profile spans
    # +/- 45 km/s), so only the first crossing is used.
    if not np.any(zs >= Z_DETECT):
        return np.inf
    i = np.argmax(zs >= Z_DETECT)
    if i == 0:
        return grid[0]
    x0, x1 = grid[i - 1], grid[i]
    y0, y1 = np.log10(zs[i - 1]), np.log10(zs[i])
    return float(x0 + (np.log10(Z_DETECT) - y0) * (x1 - x0) / (y1 - y0))


def best_line(c: Curves, inst: detect.Instrument, snr: float, log_eps: float):
    best, bz = None, -1.0
    for ln, ew_row, fw in zip(c.lines, c.ew, c.fwhm_int, strict=True):
        if not inst.wmin <= ln.wl <= inst.wmax:
            continue
        w = ew_at(ew_row, log_eps)
        z = w / detect.sigma_ew(ln.wl, inst.R, snr, detect.observed_fwhm(ln.wl, inst.R, fw, c.host.vsini))
        if z > bz:
            best, bz = ln, z
    return best, bz


__all__ = ["build_curves", "required_log_eps", "significance", "absolute_v", "Curves", "C"]
