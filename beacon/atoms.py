"""Atomic data: ionization energies, partition functions and line lists.

Ionization energies are from the NIST Atomic Spectra Database (Kramida et al. 2024).
Partition functions of Fe and Tc are summed over the levels in Kurucz's line lists;
those of Pm are tabulated by Fivet et al. (2007, Table 6). Line data for Fe and Tc are
Kurucz's (gf2600.all, gf2601.all, gf4300.pos, gf4301.pos); for Pm II they are the
HFR+CPOL oscillator strengths of Fivet et al. (2007, Table 5 and erratum).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache

import numpy as np

from . import DATA, KURUCZ
from .const import C, H, K

# First, second, third ionization energies (eV), NIST ASD; bracketed NIST values for Tc II/III, Pm II/III.
IONIZATION = {
    1: (13.598434599702,),
    26: (7.9024681, 16.19921, 30.651),
    43: (7.11938, 15.26, 29.55),
    61: (5.58187, 10.938, 22.44),
    20: (6.1131549210, 11.871719, 50.91316),
    24: (6.76651, 16.486305, 30.959),
    88: (5.2784239, 10.14718, 31.0),
}
# Atomic masses (u) of the nuclides used: Fe, Ca, Cr (natural), 99Tc, 147Pm, 226Ra.
MASS = {26: 55.845, 43: 98.906, 61: 146.915, 20: 40.078, 24: 51.996, 88: 226.025}
NAME = {26: "Fe", 43: "Tc", 61: "Pm", 20: "Ca", 24: "Cr", 88: "Ra"}

HC_K = H * C / K  # second radiation constant (cm K)


@dataclass
class Line:
    wl: float  # air wavelength (Angstrom)
    loggf: float
    elow: float  # lower-level energy (cm^-1)
    z: int  # atomic number
    ion: int  # 0 neutral, 1 singly ionized
    gamma_rad: float | None = None  # log10 Gamma_rad (s^-1)
    gamma_stark: float | None = None  # log10 Gamma_4 / n_e at 10^4 K (s^-1 cm^3)
    gamma_waals: float | None = None  # log10 Gamma_6 / n_H at 10^4 K (s^-1 cm^3)

    @property
    def label(self) -> str:
        return f"{NAME[self.z]} {'I' * (self.ion + 1)} {self.wl:.2f}"


def _kurucz_rows(path):
    for ln in path.read_text().splitlines():
        if len(ln) < 98:
            continue
        try:
            wl = float(ln[0:11])
            loggf = float(ln[11:18])
            code = float(ln[18:24])
            e1, j1 = float(ln[24:36]), float(ln[36:41])
            e2, j2 = float(ln[52:64]), float(ln[64:69])
            gr, gs, gw = float(ln[80:86]), float(ln[86:92]), float(ln[92:98])
        except ValueError:
            continue
        yield wl, loggf, code, e1, j1, e2, j2, gr, gs, gw


@cache
def kurucz_levels(fname: str) -> np.ndarray:
    """Unique (E [cm^-1], J) pairs in a Kurucz line list."""
    levels = set()
    for _, _, _, e1, j1, e2, j2, *_ in _kurucz_rows(KURUCZ / fname):
        levels.add((round(abs(e1), 2), j1))
        levels.add((round(abs(e2), 2), j2))
    return np.array(sorted(levels))


LEVEL_FILES = {(26, 0): "gf2600.all", (26, 1): "gf2601.all", (43, 0): "gf4300.pos", (43, 1): "gf4301.pos",
               (24, 1): "gf2401.all", (20, 1): "gf2001.all"}
# Ground-state statistical weights used where no level list is available.
GROUND_G = {(26, 2): 25.0, (43, 2): 6.0, (88, 2): 1.0}  # Ra III is closed-shell 6p6; NIST lists no excited level


@cache
def _ra2_rows() -> list[dict]:
    return list(csv.DictReader(ln for ln in (DATA / "ra2_nist.csv").open() if not ln.startswith("#")))


def _ra2_partition(T: np.ndarray) -> np.ndarray:
    """Ra II partition function summed over the NIST levels that appear in ra2_nist.csv."""
    levels = set()
    for r in _ra2_rows():
        levels.add((float(r["elow_cm"]), float(r["jlow"])))
        levels.add((float(r["eupp_cm"]), float(r["jupp"])))
    lev = np.array(sorted(levels))
    g = 2 * lev[:, 1] + 1
    return np.sum(g[None, :] * np.exp(-HC_K * lev[None, :, 0] / T[:, None]), axis=1)


def _ra1_partition(T: np.ndarray) -> np.ndarray:
    """Ra I partition function summed over the NIST levels in ra1_levels.csv."""
    lev = np.loadtxt(DATA / "ra1_levels.csv", delimiter=",", comments="#", skiprows=3)
    g = 2 * lev[:, 1] + 1
    return np.sum(g[None, :] * np.exp(-HC_K * lev[None, :, 0] / T[:, None]), axis=1)


@cache
def _pm_partition():
    rows = [r for r in csv.reader(ln for ln in (DATA / "pm_partition_fivet2007.csv").open() if not ln.startswith("#"))]
    arr = np.array([[float(x) for x in r] for r in rows[1:]])
    return arr


def partition(z: int, ion: int, T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if z == 1:
        return np.full_like(T, 2.0) if ion == 0 else np.ones_like(T)
    if z == 61:
        tab = _pm_partition()
        Tc = np.clip(T, tab[0, 0], tab[-1, 0])
        return np.interp(Tc, tab[:, 0], tab[:, 1 + ion])
    if (z, ion) == (88, 0):
        return _ra1_partition(T)
    if (z, ion) == (88, 1):
        return _ra2_partition(T)
    if (z, ion) in LEVEL_FILES:
        lev = kurucz_levels(LEVEL_FILES[(z, ion)])
        chi = IONIZATION[z][ion] * 8065.544  # eV -> cm^-1
        lev = lev[lev[:, 0] < chi]
        g = 2 * lev[:, 1] + 1
        return np.sum(g[None, :] * np.exp(-HC_K * lev[None, :, 0] / T[:, None]), axis=1)
    return np.full_like(T, GROUND_G[(z, ion)])


def kurucz_lines(fname: str, z: int, ion: int, wmin: float, wmax: float) -> list[Line]:
    """Lines between wmin and wmax (Angstrom) from a Kurucz file (wavelengths there are nm, air above 200 nm)."""
    out = []
    for wl, loggf, _code, e1, _j1, e2, _j2, gr, gs, gw in _kurucz_rows(KURUCZ / fname):
        if not (wmin <= 10 * wl <= wmax):
            continue
        elow = min(abs(e1), abs(e2))
        out.append(Line(10 * wl, loggf, elow, z, ion,
                        gr if gr != 0 else None, gs if gs != 0 else None, gw if gw != 0 else None))
    return out


def tc1_resonance() -> list[Line]:
    """The three Tc I resonance lines (a6S - z6P) at 4238, 4262 and 4297 Angstrom."""
    lines = kurucz_lines("gf4300.pos", 43, 0, 4200, 4310)
    return sorted([ln for ln in lines if ln.elow == 0.0], key=lambda x: x.wl)


def tc2_lines(min_loggf: float = -1.0) -> list[Line]:
    return [ln for ln in kurucz_lines("gf4301.pos", 43, 1, 2000, 10000) if ln.loggf >= min_loggf]


def pm2_lines(include_cancelled: bool = False) -> list[Line]:
    out = []
    for r in csv.DictReader(ln for ln in (DATA / "pm2_fivet2007.csv").open() if not ln.startswith("#")):
        if r["cancel"] == "1" and not include_cancelled:
            continue
        out.append(Line(10 * float(r["wl_nm"]), float(r["loggf_cpol"]), float(r["elow_cm"]), 61, 1))
    return out


def ra2_lines() -> list[Line]:
    """Ra II lines with NIST transition probabilities (air wavelengths above 2000 Angstrom)."""
    out = []
    for r in _ra2_rows():
        gl = 2 * float(r["jlow"]) + 1
        out.append(Line(10 * float(r["wl_nm"]), float(np.log10(gl * float(r["f"]))), float(r["elow_cm"]), 88, 1))
    return out


def ion_lines(fname: str, z: int, min_loggf: float = -1.0, wmin: float = 1500, wmax: float = 10000) -> list[Line]:
    """Lines of a singly ionized species from a Kurucz gfall file."""
    return [ln for ln in kurucz_lines(fname, z, 1, wmin, wmax) if ln.loggf >= min_loggf]
