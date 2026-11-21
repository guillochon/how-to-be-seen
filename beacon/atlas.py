"""Read Castelli & Kurucz (2003) ATLAS9 model atmospheres in Kurucz's text format.

A grid file (e.g. ap00k2odfnew.dat) holds many models, each starting with a TEFF line.
Each model gives, per depth, the column mass RHOX (g cm^-2), T (K), gas pressure P
(dyn cm^-2) and electron density XNE (cm^-3), plus the abundances it was computed with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import KURUCZ


@dataclass
class Model:
    teff: float
    logg: float
    vturb: float  # microturbulence the model was built with (km s^-1)
    rhox: np.ndarray
    T: np.ndarray
    P: np.ndarray
    ne: np.ndarray
    # log10 number abundance relative to H for Z = 1..99 (index 0 is H itself, set to 0)
    log_abund: np.ndarray = field(repr=False, default=None)
    he_over_h: float = 0.0851


_NUM = re.compile(r"[-+]?\d*\.\d+(?:E[-+]?\d+)?")


def _parse_block(lines: list[str]) -> Model:
    head = lines[0].split()
    teff = float(head[1])
    logg = float(head[3])
    vturb = 0.0
    for ln in lines[:3]:
        if "VTURB" in ln:
            vturb = float(ln.split("VTURB=")[1].split()[0].replace("KM/SEC", ""))
    abund = np.full(100, -20.0)
    scale = 1.0
    rows = []
    reading = False
    nrows = 0
    for ln in lines:
        if ln.startswith("ABUNDANCE SCALE"):
            scale = float(ln.split()[2])
        if "ABUNDANCE CHANGE" in ln:
            toks = ln.split("ABUNDANCE CHANGE")[1].split()
            for z, a in zip(toks[0::2], toks[1::2], strict=True):
                abund[int(z)] = float(a)
        elif ln.startswith("READ DECK6"):
            nrows = int(ln.split()[2])
            reading = True
        elif reading and len(rows) < nrows:
            # fixed-width columns can touch ("1.776E+06-3.737E-04" in the metal-poor grids)
            rows.append([float(x) for x in _NUM.findall(ln)[:4]])
    rows = np.array(rows)
    # Kurucz lists H and He as number fractions of all nuclei; metals as log10 of that fraction, to be multiplied by
    # the ABUNDANCE SCALE (0.01 in the [M/H] = -2 grids; 1 in the solar grid).
    xh, xhe = abund[1], abund[2]
    log_abund = np.full(100, -20.0)
    log_abund[0] = 0.0
    for z in range(3, 100):
        log_abund[z] = abund[z] + np.log10(scale) - np.log10(xh)
    return Model(teff, logg, vturb, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                 log_abund=log_abund, he_over_h=xhe / xh)


def read_grid(path=None) -> list[Model]:
    path = path or KURUCZ / "atm" / "ap00k2odfnew.dat"
    text = path.read_text().splitlines()
    starts = [i for i, ln in enumerate(text) if ln.startswith("TEFF")]
    starts.append(len(text))
    return [_parse_block(text[a:b]) for a, b in zip(starts[:-1], starts[1:], strict=True)]


def sun() -> Model:
    return read_grid(KURUCZ / "atm" / "asunodfnew.dat")[0]


_GRID: list[Model] | None = None


def model(teff: float, logg: float) -> Model:
    """Return the grid model at exactly (teff, logg); the grid has 250 K and 0.5 dex steps."""
    global _GRID
    if _GRID is None:
        _GRID = read_grid()
    for m in _GRID:
        if abs(m.teff - teff) < 1 and abs(m.logg - logg) < 0.01:
            return m
    raise KeyError(f"no ATLAS9 model at Teff={teff}, logg={logg}")
