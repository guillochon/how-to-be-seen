"""Upper limits on Tc in a star from its archival spectra, and what they allow a Tc beacon to cost.

    uv run python scripts/fetch_spectra.py hd101065    # once: the star's archival spectra
    uv run python scripts/tc_limits.py hd101065        # writes research/tc_limits/<star>.json and stacks
    uv run python scripts/tc_limits_plot.py hd101065   # plots in research/tc_limits/<star>/
    uv run python scripts/tc_batch.py                  # all three for every star, and a summary table

Stars are listed in STARS (coordinates, parameters and their source): hand-checked entries, and the
stars of the Ghazaryan et al. (2018) compilation selected by scripts/cp_sample.py. To add one, give its
entry and run the three commands.

Spectra. The archives are ESO (UVES, HARPS, FEROS, ESPRESSO), MAST (HST STIS and GHRS, IUE), the
Haute-Provence archive (ELODIE, SOPHIE) and PolarBase (ESPaDOnS, Narval). Each instrument's spectra
are shifted to rest with a cross-correlation against a line mask (the star's dominant species,
Star.mask), averaged with inverse-variance weights and divided by an upper envelope (the PCT
percentile of the flux in a running WIN window, smoothed). Each STIS echelle grating counts as one
instrument, its orders merged per exposure, as are the orders of ESPaDOnS and Narval. STIS, GHRS and
IUE wavelengths are converted from vacuum to air.

Limit. A Tc line cannot absorb more than all the lines at its wavelength together. In the optically
thin limit each blend component multiplies the transmission, so the Tc-only residual flux r_Tc must stay
above the observed residual r_obs (less NSIG times its noise) at every pixel within the line's core,
both measured from the highest flux within WIN/2 of the line. The largest abundance on the LOG_EPS
grid that satisfies this is the limit from that line, found by bisection, since the test only gets
harder as the abundance grows. Each line gives a valid limit by itself, and the lowest is quoted.
The Tc-only profiles come from the LTE synthesis of beacon/synth.py in the ATLAS9 model nearest the
star, broadened to the width of the star's cross-correlation peak. The model omits metal bound-free
opacity, and the star's own lines add opacity among which a Tc line forms; both weaken Tc lines, most
in the ultraviolet. To bracket this the synthesis is repeated with the continuum opacity multiplied
by each factor in BLANKET.

Validation and calibration. The same test applied to the strongest unsaturated Fe I and Fe II lines
in each instrument's range (FE_EW) must not give limits below the star's measured Fe (Star.fe). Where
the model makes lines too deep, mostly from its missing bound-free opacity in the blue and ultraviolet,
the Fe lines fail; they pass once the continuum opacity is raised enough. Each instrument's Tc limits
are therefore read at the least BLANKET factor at which its Fe lines pass (FE_OUTLIER of them may fall
more than FE_TOL below the measured Fe, for wrong gf values), and an instrument that fails at every
factor is not used: IUE high-dispersion spectra do in some stars, since their background subtraction
leaves lines too shallow. This calibrated limit is the one quoted; the limits at each factor are kept.

Inventory. The limit becomes a mass of 99Tc and an upkeep power (neutron capture, as in the paper) for
two mixed layers: the atmosphere above tau5000 = TAU_BASE, and the outer mixed zone of a normal star of
that temperature (the convection zone of Pinsonneault et al. 2001 at or below 6400 K; the turbulent
layer log(Delta M / M*) = -5.5 of an Am star above).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import maximum_filter1d, percentile_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import KURUCZ, ROOT, atlas, atoms, hosts, synth  # noqa: E402

# Velocity masks, (Kurucz species codes, lowest log gf - theta chi kept): the ionized rare earths and Fe II
# that dominate cool Ap stars (Cowley et al. 2000), and the ionized iron group that dominates hotter CP
# stars. The ultraviolet of the hotter stars is crowded, so their mask keeps weaker lines.
AP_MASK = (("6001", "5801", "5901", "6401", "2601"), -1.0)
FE_GROUP_MASK = (("2601", "2401", "2201", "2501"), -3.0)
# Normal late-B stars: Mg II and Si II, led by Mg II 4481 and Si II 6347, 6371 (11 optical lines at 10,000 K, 81 at
# 16,000 K). Their weak iron-group lines give no velocity (HD 182255: 3 sigma) or a false one that still passes
# CCF_MIN (HD 27396, HD 209008: Mg II 4481 then lies 75 km/s off in the stack). Optical spectra only: in the
# ultraviolet it would rest on the saturated Mg II 2795, 2803 lines with their interstellar components (HD 215573
# E230H: exposures 25 km/s apart), and the iron group is rich there.
B_MASK = (("1201", "1401"), -4.5)


@dataclass(frozen=True)
class Star:
    name: str
    ra: float  # ICRS (deg)
    dec: float
    teff: float
    logg: float
    radius: float  # Rsun
    fe: tuple[float, float] | None  # range of log eps(Fe) measured in the atmosphere; None if unmeasured
    source: str
    mask: tuple[tuple[str, ...], float] = AP_MASK  # velocity mask
    dilution: float = 1.0  # fraction of the near-ultraviolet continuum from this star (below 1 with a companion)
    fallback: tuple[tuple[str, ...], float] | None = None  # velocity mask for an exposure where mask gives none
    uv_mask: tuple[tuple[str, ...], float] | None = None  # velocity mask for STIS, GHRS and IUE, if not mask
    cp: str = ""  # class in the compilation or list (ApBp, HgMn, AmFm, normal, metal-poor, HB, ...)


STARS = {
    "hd101065": Star("HD 101065 (Przybylski's star)", 174.4043, -46.7097, 6400, 4.2, 1.98, (5.6, 6.5),
                     "Shulyak et al. (2010): Teff, log g and R from the energy distribution and the Hipparcos "
                     "distance; Fe an order of magnitude below solar and stratified by about 1 dex"),
    "hd9996": Star("HR 465 (HD 9996)", 24.632606, 45.399704, 10750, 4.0, 2.23, (8.3, 8.7),
                   "Nielsen et al. (2020): ATLAS9 model of the STIS (ASTRAL) phase, Fe II 8.29-8.35 and Fe I 8.73; "
                   "R from log L = 1.70 and log Teff = 4.012 of Kochukhov & Bagnulo (2006)", FE_GROUP_MASK),
    "hd175640": Star("HD 175640", 284.094416, -1.799862, 12000, 3.95, 2.82, (7.0, 7.25),
                     "Castelli & Hubrig (2004) model as used by Castelli et al. (2015), Fe 7.17-7.23 there and 7.02 in "
                     "Landstreet et al. (2009); R from Gaia DR3 (FLAME)", FE_GROUP_MASK),
    "hd141556": Star("chi Lup A (HD 141556)", 237.739772, -33.627147, 10650, 3.9, 2.85, (7.65, 7.72),
                     "Wahlgren et al. (1994): Teff, log g, Fe; Le Bouquin et al. (2013): R. chi Lup B (9200 K, 1.75 Rsun) "
                     "adds 14% of the light near 2600 A (ATLAS9 fluxes; they give L_A/L_B = 3.9 at 4400 A against "
                     "3.88 from the same models)", FE_GROUP_MASK, 0.86),
    "hd48915": Star("Sirius A (HD 48915)", 101.287155, -16.716116, 9950, 4.3, 1.711, (7.6, 7.85),
                    "Landstreet et al. (2009): Teff, log g, Fe 7.73; Kervella et al. (2003): interferometric R. "
                    "Sirius B adds well under 1% of the light near 2600 A", FE_GROUP_MASK),
    "hd72660": Star("HD 72660", 128.506726, -2.151556, 9650, 4.05, float("nan"), (7.55, 7.75),
                    "Landstreet et al. (2009): Teff, log g, Fe 7.65; no published or Gaia DR3 radius", FE_GROUP_MASK),
    "hd120709": Star("3 Cen A (HD 120709)", 207.956672, -32.994085, 17000, 3.7, 2.96, (7.7, 7.8),
                     "Landstreet et al. (2009): Teff, log g, Fe 7.80; Adelman & Pintado (2000): Fe 7.70-7.77; "
                     "R from Gaia DR3 (FLAME, with a Gaia Teff of 14,600 K)", FE_GROUP_MASK),
    "hd78316": Star("kappa Cnc (HD 78316)", 136.936706, 10.668200, 13400, 3.9, 4.87, (7.53, 7.69),
                    "Landstreet et al. (2009): Teff, log g, Fe 7.53; Adelman & Pintado (2000): Fe to 7.69; R from Gaia "
                    "DR3 (FLAME). The secondary (8500 K, R_B/R_A = 0.48; Ryabchikova et al. 1998) adds 2% near 2600 A",
                    FE_GROUP_MASK, 0.98),
    "hd193452": Star("HR 7775 (HD 193452)", 305.193950, -14.784890, 10600, 4.15, 1.98, (7.7, 7.95),
                     "Landstreet et al. (2009): Teff, log g, Fe 7.83; R from Gaia DR3 (FLAME)", FE_GROUP_MASK),
    "hd33904": Star("mu Lep (HD 33904)", 78.232929, -16.205464, 12750, 4.0, 3.39, (7.28, 7.37),
                    "Kochukhov et al. (2011): Teff, log g, R; Adelman & Pintado (2000): Fe 7.28-7.37", FE_GROUP_MASK),
    # Test cases: cool Ap stars with claims of Pm or actinides, and their analogues (Ap mask).
    "hd217522": Star("HD 217522", 345.445088, -44.840800, 6750, 4.3, 2.07, (6.4, 6.7),
                     "Gelbmann (1998): Teff, log g, Fe 6.56; R from log L = 0.85 and log Teff = 3.816 of Kochukhov & "
                     "Bagnulo (2006)"),
    "hd965": Star("HD 965", 3.516937, -0.033361, 7500, 4.0, 2.44, None,
                  "Ryabchikova et al. (2008): Teff and log g from Stromgren photometry; R from Gaia DR3 (FLAME); "
                  "no published Fe"),
    "cpd-62_2717": Star("CPD-62 2717", 185.804100, -63.275556, 6834, 3.92, 2.41, None,
                        "Gaia DR3 Teff, log g and R as quoted by Chojnowski et al. (2023); no published Fe"),
    "hd65949": Star("HD 65949", 119.448626, -60.609700, 13100, 4.0, 2.62, (7.8, 8.05),
                    "Cowley et al. (2010): Teff, log g, Fe 7.95 from Fe II and Fe III (log Fe/N_tot = -4.06, -4.09); "
                    "R from Gaia DR3 (FLAME)", FE_GROUP_MASK),
    # HgMn and Am stars of Landstreet et al. (2009), whose Teff, log g and Fe (+/- 0.15 dex) are used; R from
    # Gaia DR3 (FLAME) where it exists.
    "hd209625": Star("32 Aqr (HD 209625)", 331.197593, -0.906344, 7700, 3.75, 3.13, (7.47, 7.77),
                     "Landstreet et al. (2009): Fe 7.62; Gaia DR3 R", FE_GROUP_MASK),
    "hd27962": Star("68 Tau (HD 27962)", 66.372436, 17.927907, 8975, 4.0, float("nan"), (7.65, 7.95),
                    "Landstreet et al. (2009): Fe 7.80; no Gaia DR3 radius", FE_GROUP_MASK),
    "hd178065": Star("HD 178065", 286.787622, 0.640970, 12300, 3.6, 5.04, (7.2, 7.5),
                     "Landstreet et al. (2009): Fe 7.35; Gaia DR3 R", FE_GROUP_MASK),
    "hd186122": Star("46 Aql (HD 186122)", 295.553385, 12.193261, 12900, 3.7, 3.93, (7.8, 8.1),
                     "Landstreet et al. (2009): Fe 7.95; Gaia DR3 R", FE_GROUP_MASK),
    "hd189849": Star("15 Vul (HD 189849)", 300.275202, 27.753573, 7850, 3.7, float("nan"), (7.25, 7.55),
                     "Landstreet et al. (2009): Fe 7.40; no Gaia DR3 radius", FE_GROUP_MASK),
    "hd214994": Star("o Peg (HD 214994)", 340.439184, 29.307675, 9500, 3.62, float("nan"), (7.5, 7.8),
                     "Landstreet et al. (2009): Fe 7.65; no Gaia DR3 radius", FE_GROUP_MASK),
    "hd27295": Star("53 Tau (HD 27295)", 64.858740, 21.142305, 11900, 4.25, 2.03, (6.53, 6.83),
                    "Landstreet et al. (2009): Fe 6.68; Gaia DR3 R", FE_GROUP_MASK),
}

SPECTRA = ROOT / "research" / "spectra"
OUT = ROOT / "research" / "tc_limits"


LIST_NAMES = {"normal_lateb": "chemically normal late-B star", "uv_candidates": "star with near-UV echelle spectra",
              "uves_candidates": "star with UVES spectra below 3300 A", "mp_candidates": "slowly rotating metal-poor dwarf"}


def _catalogue_stars() -> dict[str, Star]:
    """Stars of the Ghazaryan et al. (2018) compilation and of the hand-compiled lists (research/tc_limits/cp_sample.csv,
    from scripts/cp_sample.py); scripts/fetch_spectra.py finds which have archival spectra. Fe is taken as the compiled
    value +/- its scatter (at least 0.15 dex); cool Ap stars get the rare-earth velocity mask, normal late-B stars the
    Mg II and Si II one in optical spectra (iron-group fallback), all others the iron-group one."""
    path = OUT / "cp_sample.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(path.open()):
        fe = (float(r["fe"]) - max(float(r["e_fe"] or 0), 0.15), float(r["fe"]) + max(float(r["e_fe"] or 0), 0.15))             if r["fe"] else None  # some HgMn stars have no Fe: their limits are uncalibrated
        teff = float(r["teff"])
        lst = r.get("list") or ("normal_lateb" if r["cp"] == "normal" else "")
        lateb = lst == "normal_lateb"
        mask = AP_MASK if r["cp"] == "ApBp" and teff < 9000 else B_MASK if lateb else FE_GROUP_MASK
        origin = (f"{LIST_NAMES[lst]} (research/{lst}.md, {r['cp']}): Teff, log g from {r['params_ref']}, " if lst else
                  f"Ghazaryan et al. (2018) compilation ({r['cp']}): Teff, log g, ")
        key = "".join(r["star"].lower().split())
        out[key] = Star(r["star"], float(r["ra"]), float(r["dec"]), teff, float(r["logg"]),
                        float(r["radius"]) if r["radius"] else float("nan"), fe,
                        origin + f"v sin i {r['vsini']} km/s, Fe from {r['fe_ref'] or 'unlisted'}; R from Gaia DR3 (FLAME)",
                        mask, fallback=FE_GROUP_MASK if lateb else None, uv_mask=FE_GROUP_MASK if lateb else None,
                        cp=r["cp"])
    return out


STARS = {**_catalogue_stars(), **STARS}  # the hand-checked entries above take precedence


def _radii(stars: dict[str, Star]) -> dict[str, Star]:
    """Stars with no radius get the Stefan-Boltzmann one of scripts/radii.py (research/tc_limits/radii.csv)."""
    path = OUT / "radii.csv"
    if not path.exists():
        return stars
    sb = {r["key"]: float(r["radius_cal"]) for r in csv.DictReader(path.open(encoding="utf-8"))}
    return {k: s if s.radius == s.radius or k not in sb else
            replace(s, radius=sb[k], source=s.source.replace("; R from Gaia DR3 (FLAME)", "") + RADII_SOURCE)
            for k, s in stars.items()}


RADII_SOURCE = "; R from V, parallax and Teff (scripts/radii.py)"
STARS = _radii(STARS)
# Stars whose spectra a single model atmosphere cannot describe; they get no limit.
EXCLUDED = {
    "tyc627411811": "SB2: in its ESPaDOnS spectra the narrow iron-group lines run from -24 to +54 km/s and the broad "
                    "Mg II and Si II lines from +26 to -26 km/s, in antiphase",
    "hd98088a": "SB2 (HD 98088 A and B): the spectra are of both components together",
    "hd98088b": "SB2 (HD 98088 A and B): the spectra are of both components together",
    "bd+01316": "WASP-76: a companion 0.4 arcsec away, 2.5 mag fainter, in every spectrum, and an emission-line class in SIMBAD",
}
# The rare-earth mask fails on some exposures of cool Ap stars, and on every exposure of a few (HD 116235, 64 Vir:
# 42 spectra, none with a velocity; the iron-group mask gives -10 km/s on each). Their iron group is the fallback.
STARS = {k: replace(s, fallback=FE_GROUP_MASK) if s.mask is AP_MASK and s.fallback is None else s for k, s in STARS.items()}
RSUN_CM, MSUN_G, G_CGS, CKMS = 6.957e10, 1.989e33, 6.674e-8, 2.998e5
WIN, PCT = 10.0, 97.0  # envelope window (A) and percentile
NSIG = 3.0
CCF_MIN = 4.0  # least cross-correlation peak, in units of its scatter away from the peak
CCF_WIN = 120.0  # km/s either side of the peak over which its significance is measured
VMAX = 450.0  # km/s searched for a velocity: halo stars reach 364 (HD 161817) and 205 (HD 93329), plus 30 for the Earth
BLANKET = (1.0, 3.0, 10.0)
LOG_EPS = np.arange(-2.0, 8.01, 0.1)  # above about 8 a Tc line saturates and the test is no longer monotonic
DV = 0.25  # km/s, sampling of the synthetic profiles
TAU_BASE = 3.0
EW_MIN = 5.0  # mA at log eps = 3, the weakest Tc line kept
PRE_CUT = {0: -1.6, 1: -3.4}  # log gf - theta chi below which a Tc I / Tc II line cannot reach EW_MIN
N_TC_MIN = 10  # fewest Tc lines tested
N_FE = 60  # unsaturated Fe I and Fe II lines used to test the method in each instrument
FE_EW = (5.0, 60.0)  # mA at the measured Fe: the unsaturated lines used to test the method
FE_SCAN = 1500  # strongest Fe lines examined per instrument to find N_FE unsaturated ones
FE_TOL = 0.3  # dex an Fe limit may fall below the measured Fe and still pass
FE_OUTLIER = 0.1  # fraction of an instrument's Fe lines allowed to fail (wrong gf values, misidentified lines)
VSTEP = {"uves": 1.5, "harps": 0.8, "feros": 2.0, "espresso": 0.5, "elodie": 2.0, "sophie_hr": 1.0, "sophie_he": 2.0, "espadons": 1.5, "narval": 1.5, "iue": 5.0, "e230h": 1.0, "e230m": 3.0, "ghrs_echb": 1.0, "ghrs_g270m": 3.0}  # km/s, stacked pixel


def vac_to_air(w):
    """Air wavelengths (A) from vacuum ones above 2000 A (Morton 2000)."""
    s2 = (1e4 / w) ** 2
    n = 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)
    return np.where(w > 2000, w / n, w)


def read_eso(path: Path):
    """(air wavelength A, flux, error) of an ESO Phase 3 1D spectrum."""
    with fits.open(path) as h:
        d = h[1].data
        names = d.columns.names
        w = np.asarray(d["WAVE_AIR"][0] if "WAVE_AIR" in names else d["WAVE"][0], float)  # ESPRESSO: vacuum WAVE
        f = np.asarray(d["FLUX"][0] if "FLUX" in names else d["FLUX_REDUCED"][0], float)
        e = np.asarray(d["ERR"][0] if "ERR" in names else d["ERR_REDUCED"][0], float)
        snr = h[0].header.get("SNR")
    if not np.isfinite(e).any():
        # Older HARPS products carry no error array: photon noise scaled to the header S/N near 5500 A.
        fref = np.nanmedian(f[(w > 5450) & (w < 5550)])
        e = np.sqrt(np.clip(f, 0, None) * fref) / snr
    return w, f, e


def read_elodie(path: Path):
    """(air wavelength A, flux, error) of an ELODIE 1D spectrum (s1d) with its NOISE extension."""
    with fits.open(path) as h:
        hd = h[0].header
        f = np.asarray(h[0].data, float)
        e = np.asarray(h["NOISE"].data, float)
    w = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(len(f)) + 1 - hd["CRPIX1"])
    return w, f, e


def read_sophie(path: Path):
    """(air wavelength A, flux, error) of a SOPHIE 1D spectrum (s1d). It carries no error array: photon noise
    is scaled to the S/N of order 26 (near 5500 A) from the header; None if the header has none (some of theta Cyg's)."""
    with fits.open(path) as h:
        hd = h[0].header
        f = np.asarray(h[0].data, float)
    w = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(len(f)) + 1 - hd["CRPIX1"])
    snr = hd.get("HIERARCH OHP DRS SPE EXT SN26") or hd.get("HIERARCH OHP DRS CAL EXT SN26")
    if not snr:
        return None
    fref = np.nanmedian(f[(w > 5450) & (w < 5550)])
    e = np.sqrt(np.clip(f, 1e-6 * fref, None) * fref) / snr
    return w, f, e


def read_iue(path: Path):
    """(wavelength A, flux, error) of the IUE high-dispersion orders, each trimmed to its usable part."""
    with fits.open(gzip.open(path)) as h:
        d = h[1].data
        out = []
        for row in d:
            i0, n = int(row["STARTPIX"]) - 1, int(row["NPOINTS"])
            sl = slice(i0, i0 + n)
            w = vac_to_air(row["WAVELENGTH"] + row["DELTAW"] * np.arange(n))
            f = np.asarray(row["ABS_CAL"][sl], float)
            net = np.asarray(row["NET"][sl], float)
            e = np.asarray(row["NOISE"][sl], float) * np.abs(f / np.where(net != 0, net, np.nan))
            ok = (np.asarray(row["QUALITY"][sl]) == 0) & np.isfinite(e) & (f > 0)
            if ok.sum() > 100:
                out.append((w[ok], f[ok], e[ok]))
    return out


def merge_orders(orders):
    """One (w, f, e) from overlapping echelle orders, averaged with inverse-variance weights on a log-lambda
    grid as fine as the finest order."""
    lo, hi = min(o[0][0] for o in orders), max(o[0][-1] for o in orders)
    step = min(np.median(np.diff(o[0])) / o[0].mean() for o in orders)
    grid = np.exp(np.arange(np.log(lo), np.log(hi), step))
    num, den = np.zeros_like(grid), np.zeros_like(grid)
    for w, f, e in orders:
        inside = (grid >= w[0]) & (grid <= w[-1])
        num[inside] += np.interp(grid[inside], w, f) / np.interp(grid[inside], w, e) ** 2
        den[inside] += 1 / np.interp(grid[inside], w, e) ** 2
    ok = den > 0
    return grid[ok], num[ok] / den[ok], 1 / np.sqrt(den[ok])


def read_polarbase(path: Path):
    """(wavelength A, normalized intensity, error) of an ESPaDOnS or Narval spectrum from PolarBase (.s, the
    Libre-ESpRIT text format: wavelength in nm, intensity, [Stokes V and two null profiles,] error) or of an
    ESPaDOnS spectrum from CADC (Upena i.fits: its first three rows are the same wavelength, intensity and error,
    as for HD 22136 exposure 2335352), its overlapping orders merged."""
    if path.suffix == ".fits":
        d = fits.getdata(path).astype(float)
        w, f, e = 10 * d[0], d[1], d[2]
    else:
        d = np.loadtxt(path, skiprows=2)
        w, f, e = 10 * d[:, 0], d[:, 1], d[:, -1]
    cuts = np.flatnonzero(np.diff(w) < 0) + 1  # a new order starts where the wavelength falls back
    orders = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(w)], strict=True):
        ok = (e[a:b] > 0) & np.isfinite(f[a:b])
        if ok.sum() > 100:
            orders.append((w[a:b][ok], f[a:b][ok], e[a:b][ok]))
    return merge_orders(orders)


def read_stis(path: Path):
    """(grating, (air wavelength A, flux, error)) of an HST STIS echelle exposure (x1d), its orders merged
    with inverse-variance weights. Pixels flagged in SDQFLAGS are dropped."""
    with fits.open(path) as h:
        grating = h[0].header["OPT_ELEM"].lower()
        orders = []
        for ext in h[1:]:
            if ext.header.get("EXTNAME") != "SCI":
                continue
            bad = ext.header.get("SDQFLAGS", 31743)
            for row in ext.data:
                w, f, e, dq = (np.asarray(row[k], float) for k in ("WAVELENGTH", "FLUX", "ERROR", "DQ"))
                ok = ((dq.astype(int) & bad) == 0) & (e > 0) & np.isfinite(f)
                if ok.sum() > 100:
                    orders.append((vac_to_air(w[ok]), f[ok], e[ok]))
    return grating, merge_orders(orders)


def read_ghrs(c1f: Path):
    """(grating, (air wavelength A, flux, error)) of an HST GHRS exposure from its calibrated c0f
    (wavelength), c1f (flux) and c2f (error) files; each group (sub-exposure) is kept separately."""
    stem = str(c1f)[: -len("_c1f.fits")]
    with fits.open(stem + "_c0f.fits") as h0, fits.open(c1f) as h1, fits.open(stem + "_c2f.fits") as h2:
        grating = h1[0].header["GRATING"].lower().replace("-", "")
        w, f, e = (np.atleast_2d(np.asarray(h[0].data, float)) for h in (h0, h1, h2))
    out = []
    for wi, fi, ei in zip(w, f, e, strict=True):
        ok = (ei > 0) & np.isfinite(fi) & (wi > 0)
        if ok.sum() > 100:
            out.append((vac_to_air(wi[ok]), fi[ok], ei[ok]))
    return f"ghrs_{grating}", out


def line_mask(wmin: float, wmax: float, mask=AP_MASK, teff: float = 6500.0):
    """Air wavelengths and weights of the strongest lines of a velocity mask at temperature teff."""
    wl, wt = _mask_lines(*mask, 500 * round(teff / 500))
    sel = (wl >= wmin) & (wl <= wmax)
    return wl[sel], wt[sel]


@cache
def _mask_lines(species: tuple[str, ...], thr: float, teff: float) -> tuple[np.ndarray, np.ndarray]:
    wl, wt = [], []
    for code in species:
        for w, loggf, _c, e1, _j1, e2, *_ in atoms._kurucz_rows(ROOT / "research" / "kurucz" / "gfall" / f"gf{code}.all"):
            s = loggf - 5040 / teff * min(abs(e1), abs(e2)) / 8065.54
            if s > thr and w > 200:
                wl.append(10 * w)
                wt.append(s - thr + 1)
    o = np.argsort(wl)
    return np.array(wl)[o], np.clip(np.array(wt)[o], 0.1, None)


def ccf(w, r, mask_wl, mask_wt, vgrid):
    """Cross-correlation of the absorption 1 - r with a weighted line mask."""
    a = 1.0 - r
    return np.array([np.sum(mask_wt * np.interp(mask_wl * (1 + v / 2.998e5), w, a, left=0, right=0)) for v in vgrid])


def envelope(w, f):
    """Upper envelope: the PCT percentile of f in a running WIN window, smoothed over the same window."""
    step = np.median(np.diff(w))
    n = max(int(WIN / step), 5)
    e = percentile_filter(f, PCT, size=n, mode="nearest")
    e = maximum_filter1d(e, size=max(n // 5, 3), mode="nearest")
    return uniform_filter1d(e, size=n, mode="nearest")


def _velocity(w, r, mask, teff: float, edge: float, vgrid):
    """(velocity, CCF FWHM) of one normalized exposure, or None if the CCF has no clear peak."""
    mw, mt = line_mask(w[0] + edge, w[-1] - edge, mask, teff)
    if not len(mw):
        return None
    c = ccf(w, r, mw, mt, vgrid)
    i = int(np.argmax(c))
    half = c[i] - 0.5 * (c[i] - np.median(c))
    j0, j1 = i, i
    while j0 > 0 and c[j0 - 1] > half:
        j0 -= 1
    while j1 < len(c) - 1 and c[j1 + 1] > half:
        j1 += 1
    # Peak height against the scatter away from the peak, so broad-lined stars are not penalized, within CCF_WIN of
    # it, so the test does not depend on how wide a range is searched (with all of +-VMAX, L Car lost three of its
    # five UVES exposures).
    dv = np.abs(vgrid - vgrid[i])
    off = c[(dv > 1.5 * (vgrid[j1] - vgrid[j0])) & (dv < CCF_WIN)]
    if not 0 < i < len(c) - 1 or len(off) < 40 or not (c[i] - np.median(off)) / np.std(off) >= CCF_MIN:
        return None
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    return vgrid[i] + (vgrid[1] - vgrid[0]) * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2), vgrid[j1] - vgrid[j0]


def stack(spectra, vstep: float, mask=(AP_MASK, 6500.0), vel: float | None = None, edge: float = 5.0, fallback=None):
    """Shift each (w, f, e) to rest with its CCF velocity (or vel), normalize and average on a log-lambda
    grid; an exposure whose CCF has no clear peak with mask is tried with the fallback mask, then dropped, and so is
    one whose velocity lies far outside a tight cluster of the others (an outlier, see _outliers).
    Returns (w, r, sigma, velocities, CCF FWHM km/s)."""
    lo = min(s[0][0] for s in spectra)
    hi = max(s[0][-1] for s in spectra)
    grid = np.exp(np.arange(np.log(lo), np.log(hi), vstep / 2.998e5))
    num, den = np.zeros_like(grid), np.zeros_like(grid)
    vgrid = np.arange(-VMAX, VMAX + 0.01, 0.25)
    kept = []
    for w, f, e in spectra:
        ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0)
        w, f, e = w[ok], f[ok], e[ok]
        r = f / envelope(w, f)
        if vel is None:
            peak = _velocity(w, r, mask[0], mask[1], edge, vgrid)
            if peak is None and fallback:
                peak = _velocity(w, r, fallback, mask[1], edge, vgrid)
            if peak is None:
                continue  # no clear velocity
        else:
            peak = (vel, np.nan)
        kept.append((w, f, e, r, *peak))
    bad = _outliers([k[4] for k in kept], [k[5] for k in kept]) if vel is None else set()
    vels, widths = [], []
    for n, (w, f, e, r, v, width) in enumerate(kept):
        if n in bad:
            continue
        vels.append(v)
        widths.append(width)
        wr = w / (1 + v / 2.998e5)
        inside = (grid >= wr[0]) & (grid <= wr[-1])
        rr = np.interp(grid[inside], wr, r)
        ee = np.interp(grid[inside], wr, e / envelope(w, f))
        num[inside] += rr / ee**2
        den[inside] += 1 / ee**2
    ok = den > 0
    return grid[ok], num[ok] / den[ok], 1 / np.sqrt(den[ok]), np.array(vels), np.array(widths)


def _outliers(vels: list[float], widths: list[float]) -> set[int]:
    """Exposures whose velocity is more than 3 CCF widths and 5 robust sigmas from the median of at least three: a
    noise peak (HD 338529: one UVES exposure at -441 km/s, the other four at -143 to -149). The robust sigma keeps
    every exposure of a binary whose velocities spread widely (kappa Cnc)."""
    if len(vels) < 3:
        return set()
    v = np.array(vels)
    med = np.median(v)
    cut = max(5 * 1.4826 * np.median(np.abs(v - med)), 3 * np.nanmedian(widths))
    return {n for n, x in enumerate(v) if abs(x - med) > cut}


def tc_lines(s: synth.Synth, teff: float, ranges) -> list[atoms.Line]:
    """Tc I and Tc II lines between 1850 and 7000 A with an equivalent width above EW_MIN at log eps = 3; in
    any wavelength range (wmin, wmax) holding fewer than N_TC_MIN of them, its N_TC_MIN strongest lines (in
    the hottest stars Tc is mostly Tc III)."""
    lines = atoms.kurucz_lines("gf4300.pos", 43, 0, 1850, 7000) + atoms.kurucz_lines("gf4301.pos", 43, 1, 1850, 7000)
    lines = [ln for ln in lines if ln.loggf - 5040 / teff * ln.elow / 8065.54 > PRE_CUT[ln.ion]]
    ew = np.array([s.ew(ln, 3.0) for ln in lines])
    keep = {i for i in range(len(lines)) if ew[i] > EW_MIN}
    for lo, hi in ranges:
        inside = [i for i in np.argsort(-ew) if lo + 2 <= lines[i].wl <= hi - 2]
        if len(keep.intersection(inside)) < N_TC_MIN:
            keep.update(inside[:N_TC_MIN])
    return [lines[i] for i in sorted(keep)]


def fe_lines(s: synth.Synth, teff: float, lo: float, hi: float, log_eps: float) -> list[atoms.Line]:
    """The N_FE strongest unsaturated Fe I and Fe II lines between lo and hi (A): ranked by log gf - theta chi
    (Fe II favoured by 2 dex for ionization), keeping those whose equivalent width at log_eps lies in FE_EW.
    Saturated lines form high in the atmosphere, where Fe is stratified in Ap stars and LTE cores fail; a Tc
    line at its limit is weak, so the test is made in that regime."""
    th = 5040 / teff
    lines = [ln for ln in atoms.kurucz_lines("gfall/gf2600.all", 26, 0, max(lo + 2, 3000), min(hi - 2, 6900))
             + atoms.kurucz_lines("gfall/gf2601.all", 26, 1, max(lo + 2, 2400), min(hi - 2, 6900))
             if ln.loggf - th * ln.elow / 8065.54 > (-3.0 if ln.ion == 0 else -5.0)]
    out = []
    for ln in sorted(lines, key=lambda ln: -(ln.loggf - th * ln.elow / 8065.54 + 2 * ln.ion))[:FE_SCAN]:
        if FE_EW[0] < s.ew(ln, log_eps) < FE_EW[1]:
            out.append(ln)
            if len(out) == N_FE:
                break
    return out


class Profiles:
    """Residual flux of one trace line on a velocity grid, Gaussian-broadened to fwhm_kms, at any
    abundance; the abundance-independent synthesis is done once (synth.LineFamily)."""

    def __init__(self, s: synth.Synth, line: atoms.Line, fwhm_kms: float):
        hw = 6 * fwhm_kms
        self.v = np.arange(-hw, hw + DV / 2, DV)
        self.fam = s.family(line, line.wl * (1 + self.v / CKMS))
        sig = fwhm_kms / 2.3548 / DV
        k = np.exp(-0.5 * (np.arange(-int(4 * sig) - 1, int(4 * sig) + 2) / sig) ** 2)
        self.kernel = k / k.sum()

    def __call__(self, log_eps: float) -> np.ndarray:
        return 1 - np.convolve(1 - self.fam(log_eps), self.kernel, mode="same")


def line_limit(w, r, sig, line, prof: Profiles, fwhm_kms, grid=LOG_EPS, dil=1.0):
    """Largest abundance on grid whose profile stays above r - NSIG sigma at every pixel within +/- FWHM/2.
    A companion's light (dil < 1) makes the line shallower by the factor dil."""
    vo = (w / line.wl - 1) * CKMS
    core = np.abs(vo) <= 0.5 * fwhm_kms
    if core.sum() < 2:
        return None
    # The envelope can run below the local flux peaks, so depths are measured from the highest flux
    # within +/- WIN/2: the line's continuum is never placed below a point the star reaches.
    cont = r[np.abs(w - line.wl) <= WIN / 2].max()
    floor = (r[core] - NSIG * sig[core]) / cont

    def ok(i):
        return bool(np.all(1 - dil * (1 - np.interp(vo[core], prof.v, prof(grid[i]))) >= floor))

    lo, hi = 0, len(grid) - 1
    if not ok(lo):
        return round(float(grid[0]), 1)
    if ok(hi):
        return float("inf")
    while hi - lo > 1:  # ok(lo) and not ok(hi)
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return round(float(grid[lo]), 1)


def atmosphere_mass(s: synth.Synth, radius: float) -> float:
    """Mass above tau5000 = TAU_BASE (Msun)."""
    col = float(np.interp(np.log10(TAU_BASE), np.log10(s.tau_cont_500()), s.m.rhox))
    return 4 * np.pi * (radius * RSUN_CM) ** 2 * col / MSUN_G


def outer_mixed_mass(star: Star) -> tuple[str, float]:
    """(label, Msun) of the outer mixed zone of a normal star at this temperature."""
    if star.teff <= 6400:
        return "convection zone of a normal star", hosts.pinsonneault_mcz(max(star.teff, 4000))
    mstar = 10**star.logg * (star.radius * RSUN_CM) ** 2 / G_CGS / MSUN_G
    return "turbulent layer of an Am star", hosts._a_star_mix(mstar)[0]


# Hand-checked HgMn stars. With the normal late-B stars and the HgMn stars of the compilation they are the quiet
# late-B hosts of the paper, which mix nothing below the atmosphere; every other star mixes its outer zone.
QUIET_HAND = {"hd175640", "hd178065", "hd141556", "hd120709", "hd27295", "hd186122", "hd78316", "hd33904",
              "hd193452", "hd65949"}


# Hand-checked magnetic Ap stars, which the compilation does not type.
AP_HAND = {"hd101065", "hd9996", "hd217522", "hd965", "cpd-62_2717"}


def host_class(key: str, star: Star) -> str:
    """The paper's host class whose mixed mass applies, by the star's type rather than its temperature: quiet late-B
    (atmosphere), A star (Am and Ap stars of any Teff: the turbulent layer), HB (horizontal-branch stars: the same
    layer, uncalibrated for them) or FGK (the convection zone; metal-poor dwarfs of any Teff)."""
    if key in QUIET_HAND or star.cp == "HgMn" or "normal_lateb" in star.source:
        return "quiet late-B"
    if star.cp == "HB":
        return "HB"
    if star.cp in ("ApBp", "AmFm") or key in AP_HAND:
        return "A star"
    if star.cp == "metal-poor":
        return "FGK"
    return "FGK" if star.teff <= 6400 else "A star"


def host_mixed_mass(key: str, star: Star, mixed: dict[str, float]) -> float:
    """The mixed mass (Msun) of the star's host class. The convection zone is that of a solar-metallicity star at the
    star's Teff (Pinsonneault et al. 2001), held at its 6400 K value for the metal-poor dwarfs above 6400 K."""
    cls = host_class(key, star)
    if cls == "quiet late-B":
        return next(v for m, v in mixed.items() if m.startswith("atmosphere"))
    if cls == "FGK":
        return hosts.pinsonneault_mcz(min(max(star.teff, 4000), 6400))
    mstar = 10**star.logg * (star.radius * RSUN_CM) ** 2 / G_CGS / MSUN_G
    return hosts._a_star_mix(mstar)[0]


def inventory(le: float, mmix: float) -> dict:
    m = hosts.isotope_mass(le, 98.906, mmix)
    p = hosts.upkeep_power(m, "99Tc capture") / 1e7
    return dict(mass_kg=m / 1e3, upkeep_W=p, sagan_K=(np.log10(p) - 6) / 10)


def load(key: str):
    """Stacked, normalized spectra per instrument: {name: (w, r, sigma, velocities, FWHM km/s)}."""
    spec, star = SPECTRA / key, STARS[key]
    mask = (star.mask, star.teff)
    uv = (star.uv_mask or star.mask, star.teff)
    out = {}
    readers = dict(uves=read_eso, harps=read_eso, feros=read_eso, espresso=read_eso, elodie=read_elodie,
                   sophie_hr=read_sophie, sophie_he=read_sophie, espadons=read_polarbase, narval=read_polarbase)
    for name, reader in readers.items():
        files = sorted(f for f in (spec / name).glob("*") if f.suffix in (".fits", ".s"))
        spectra = [sp for sp in (reader(f) for f in files) if sp is not None]
        if spectra:
            out[name] = stack(spectra, VSTEP[name], mask, fallback=star.fallback)
    stis = {}
    for f in sorted((spec / "stis").glob("*_x1d.fits")):
        grating, sp = read_stis(f)
        stis.setdefault(grating, []).append(sp)
    for grating, sp in stis.items():
        out[grating] = stack(sp, VSTEP[grating], uv)
    ghrs = {}
    for f in sorted(f for f in (spec / "ghrs").glob("*_c1f.fits") if Path(str(f).replace("_c1f", "_c2f")).exists()):
        grating, sp = read_ghrs(f)
        ghrs.setdefault(grating, []).extend(sp)
    for grating, sp in ghrs.items():
        # A GHRS exposure spans only 10-40 A, so its velocity uses mask lines up to 0.5 A from its ends.
        out[grating] = stack(sp, VSTEP[grating], uv, edge=0.5)
    orders = [o for f in sorted((spec / "iue").glob("*.mxhi.gz")) for o in read_iue(f)]
    if orders:
        # IUE orders are too short for a velocity each: one velocity for the merged exposure(s).
        w, r, *_ = stack(orders, VSTEP["iue"], uv, vel=0.0)
        peak = _velocity(w, r, uv[0], uv[1], 5.0, np.arange(-VMAX, VMAX + 0.01, 1.0))
        if peak:  # HD 181470 (3.9 sigma, -90 km/s against +11 in SIMBAD) and kappa Cnc (3.1 sigma) have none
            w, r, sig, _, _ = stack(orders, VSTEP["iue"], uv, vel=peak[0])
            out["iue"] = (w, r, sig, np.array([peak[0]]), np.array([peak[1]]))
    return {k: v for k, v in out.items() if len(v[3])}  # instruments with at least one usable spectrum


def limits(data, s_by_blanket, lines, grid=LOG_EPS, dil=1.0):
    """Per-line limits for every instrument and blanketing factor."""
    rows = []
    for inst, (w, r, sig, _vels, widths) in data.items():
        fwhm = float(np.nanmedian(widths))
        for ln in lines:
            if not w[0] + 2 <= ln.wl <= w[-1] - 2:
                continue
            row = dict(species=f"{atoms.NAME[ln.z]} {'I' * (ln.ion + 1)}", wl=ln.wl, loggf=ln.loggf, elow=ln.elow,
                       instrument=inst, fwhm_kms=fwhm)
            for b, s in s_by_blanket.items():
                row[f"limit_b{b:g}"] = line_limit(w, r, sig, ln, Profiles(s, ln, fwhm), fwhm, grid, dil)
            rows.append(row)
    return rows


# ATLAS9 grids of Castelli & Kurucz by [M/H] (scripts/fetch_atlas_grids.py); alpha-enhanced at -1 and below, as in
# halo stars. Only metal-poor and horizontal-branch stars use them; in CP stars a low Fe is a surface anomaly
# (Przybylski's star), so they, and any star without Fe, use the solar grid.
GRIDS = {0.0: "ap00k2odfnew", -0.5: "am05k2odfnew", -1.0: "am10ak2odfnew", -1.5: "am15ak2odfnew",
         -2.0: "am20ak2odfnew", -2.5: "am25ak2odfnew"}
FE_SUN = 7.50


def grid_name(star: Star) -> str:
    """The grid nearest the star's [Fe/H] (its Fe range's middle minus FE_SUN), solar for metal-rich stars."""
    feh = (sum(star.fe) / 2 - FE_SUN) if star.fe and star.cp in ("metal-poor", "HB") else 0.0
    return GRIDS[min(GRIDS, key=lambda m: abs(m - min(feh, 0.0)))]


@cache
def _grid(name: str) -> list:
    return atlas.read_grid(KURUCZ / "atm" / f"{name}.dat")


def synths(star: Star) -> dict[float, synth.Synth]:
    """One synthesis per BLANKET factor in the ATLAS9 model nearest the star (250 K steps to 13,000 K, then
    1000 K; 0.5 dex in log g), in the grid nearest its [Fe/H]."""
    logg = 0.5 * round(star.logg / 0.5)
    model = min((m for m in _grid(grid_name(star)) if abs(m.logg - logg) < 0.01), key=lambda m: abs(m.teff - star.teff))
    out = {}
    for b in BLANKET:
        s = synth.Synth(model, vturb=1.0)
        if b != 1:
            kc = s.st.kappa_cont
            s.st.kappa_cont = lambda wl, kc=kc, b=b: b * kc(wl)
        out[b] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("star", choices=sorted(STARS))
    key = ap.parse_args().star
    if key in EXCLUDED:
        sys.exit(f"{key}: excluded, {EXCLUDED[key]}")
    star = STARS[key]
    t0 = time.time()
    (OUT / key).mkdir(parents=True, exist_ok=True)
    data = load(key)
    if not data:
        sys.exit(f"{key}: no usable spectra")
    for k, (w, _r, sig, vels, widths) in data.items():
        print(f"{k}: {len(vels)} spectra, {w[0]:.0f}-{w[-1]:.0f} A, v = {np.median(vels):.2f} +/- {np.std(vels):.2f} km/s, "
              f"CCF FWHM {np.nanmedian(widths):.1f} km/s, median S/N {np.median(1 / sig):.0f}", flush=True)
    print(f"stacked in {time.time() - t0:.0f} s", flush=True)
    s_by_blanket = synths(star)
    s1 = s_by_blanket[1.0]
    tc = limits(data, s_by_blanket, tc_lines(s1, star.teff, [(v[0][0], v[0][-1]) for v in data.values()]),
                dil=star.dilution)
    print(f"Tc limits in {time.time() - t0:.0f} s", flush=True)
    # Without a measured Fe the test still runs (on a wide grid) but cannot reject an instrument.
    fe_lo, fe_hi = star.fe or (3.5, 8.0)
    fe_mid = 0.5 * (fe_lo + fe_hi) if star.fe else 7.0  # lines chosen as unsaturated at about solar Fe if unmeasured
    fe = [r_ for k, v in data.items() for r_ in limits({k: v}, s_by_blanket, fe_lines(s1, star.teff, v[0][0], v[0][-1], fe_mid),
                                                     np.arange(fe_lo - 4, fe_hi + 0.51, 0.1), star.dilution)]
    # Calibration: each instrument's limits are read at the least BLANKET factor at which its Fe lines pass
    # (at most FE_OUTLIER of them below the measured Fe by more than FE_TOL); an instrument that fails at
    # every factor is not used. Without a measured Fe every instrument is read at factor 1.
    calib = {}
    for k in data:
        for b in BLANKET:
            x = [r_[f"limit_b{b:g}"] for r_ in fe if r_["instrument"] == k and r_[f"limit_b{b:g}"] is not None]
            if star.fe is None or sum(v < fe_lo - FE_TOL - 1e-6 for v in x) <= FE_OUTLIER * len(x):
                calib[k] = b
                break
    trusted = sorted(calib)
    best = {}
    for b in BLANKET:
        k = f"limit_b{b:g}"
        vals = [(r_[k], r_) for r_ in tc if r_[k] is not None and np.isfinite(r_[k]) and r_["instrument"] in trusted]
        if vals:
            le, row = min(vals, key=lambda t: t[0])
            best[f"{b:g}"] = dict(log_eps=le, line=f"{row['species']} {row['wl']:.2f}", instrument=row["instrument"])
    vals = [(r_[f"limit_b{calib[r_['instrument']]:g}"], r_) for r_ in tc if r_["instrument"] in calib]
    vals = [(v, r_) for v, r_ in vals if v is not None and np.isfinite(v)]
    if vals:
        le, row = min(vals, key=lambda t: t[0])
        best["calibrated"] = dict(log_eps=le, line=f"{row['species']} {row['wl']:.2f}", instrument=row["instrument"],
                                  blanket=calib[row["instrument"]])
    outer_name, outer = outer_mixed_mass(star)
    mixed = {f"atmosphere (tau5000 < {TAU_BASE:g})": atmosphere_mass(s1, star.radius), outer_name: outer}
    out = dict(star=star.__dict__, model=dict(teff=s1.m.teff, logg=s1.m.logg, grid=grid_name(star)), nsig=NSIG,
               spectra={k: dict(n=len(v[3]), wmin=float(v[0][0]), wmax=float(v[0][-1]), v_kms=float(np.median(v[3])),
                                fwhm_kms=float(np.nanmedian(v[4])), median_snr=float(np.median(1 / v[2])))
                        for k, v in data.items()},
               tc_lines=tc, fe_lines=fe, trusted=trusted, calibration=calib, best=best, mixed_msun=mixed,
               inventory={f"{m}|b{b}": inventory(best[b]["log_eps"], mm) for m, mm in mixed.items() for b in best})
    (OUT / f"{key}.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(OUT / key / "stacks.npz", **{f"{k}_{n}": v[i] for k, v in data.items()
                                                      for i, n in enumerate(("w", "r", "sig"))})
    for r_ in sorted((r_ for r_ in tc if r_["limit_b1"] is not None), key=lambda r_: r_["limit_b1"])[:15]:
        print(f"{r_['species']:6s} {r_['wl']:9.3f} {r_['instrument']:5s} limit " +
              ", ".join(f"x{b:g} {r_[f'limit_b{b:g}'] or float('nan'):5.2f}" for b in BLANKET))
    fe_lim = np.array([r_["limit_b1"] for r_ in fe if r_["limit_b1"] is not None] or [np.nan])
    print(f"Fe test ({len(fe_lim)} lines, measured log eps {star.fe or 'unknown'}): "
          f"{np.sum(fe_lim < fe_lo - 1e-6) if star.fe else '-'} limits below the range, lowest {fe_lim.min():.2f}, median {np.median(fe_lim):.2f}; "
          f"opacity factor by instrument: {calib}")
    print(json.dumps(dict(best=best, mixed=mixed, inventory=out["inventory"]), indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
