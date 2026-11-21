"""LTE line-formation model for isotope-pollution beacons in stellar photospheres.

Modules:
    const    physical constants (cgs)
    atlas    reader for Castelli & Kurucz ATLAS9 model atmospheres
    atoms    ionization energies, partition functions, line lists (Kurucz, Fivet et al. 2007)
    opacity  continuous opacity (H-, H I, Rayleigh, Thomson)
    synth    LTE formal solution and equivalent widths
    detect   equivalent-width detection limits and photon budgets
    hosts    host-star grid, mixed masses, isotope masses and costs
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
KURUCZ = ROOT / "research" / "kurucz"
