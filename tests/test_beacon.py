"""Checks on the line-formation and detectability code in beacon/."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beacon import atlas, atoms, detect, hosts, synth  # noqa: E402

KURUCZ_OK = (ROOT / "research" / "kurucz" / "atm" / "asunodfnew.dat").exists()
needs_kurucz = pytest.mark.skipif(not KURUCZ_OK, reason="Kurucz inputs not downloaded (research/kurucz)")


@needs_kurucz
def test_solar_continuum_temperature():
    s = synth.Synth(atlas.sun())
    t5 = s.tau_cont_500()
    T1 = np.interp(0.0, np.log10(t5), s.m.T)
    assert 6300 < T1 < 6600  # standard solar models put T(tau_500 = 1) near 6400-6500 K


@needs_kurucz
def test_weak_lines_scale_linearly_with_abundance():
    s = synth.Synth(atlas.sun())
    ln = atoms.tc1_resonance()[-1]
    w1, w2 = s.ew(ln, -1.0), s.ew(ln, -0.0)
    assert w2 / w1 == pytest.approx(10.0, rel=0.05)


@needs_kurucz
def test_inverse_abundance_solver():
    s = synth.Synth(atlas.sun())
    ln = atoms.tc1_resonance()[-1]
    le = s.log_eps_for_ew(ln, 5.0)
    assert s.ew(ln, le) == pytest.approx(5.0, rel=0.01)


def test_pm_partition_matches_fivet_table():
    assert atoms.partition(61, 1, np.array([6500.0]))[0] == pytest.approx(229.85)


@needs_kurucz
def test_saha_two_stage_by_hand():
    """Pm I/II at 6500 K, P_e = 40 dyn cm^-2, against the Saha equation evaluated by hand."""
    from beacon.const import KEV, SAHA, K
    from beacon.opacity import State
    T, pe = 6500.0, 40.0
    ne = pe / (K * T)
    m0 = atlas.sun()
    m = atlas.Model(6500, 4.0, 2.0, np.array([1.0]), np.array([T]), np.array([1e5]), np.array([ne]),
                    m0.log_abund, m0.he_over_h)
    f = State(m).ion_fractions(61, atoms.IONIZATION[61], atoms.partition)[:, 0]
    u1, u2 = 145.49, 229.85  # Fivet et al. (2007) Table 6
    r = 2 * u2 / u1 * SAHA * T**1.5 * np.exp(-5.58187 / (KEV * T)) / ne
    assert f[1] / f[0] == pytest.approx(r, rel=1e-6)


def test_sigma_ew_matched_filter():
    # Gaussian FWHM w, pixel dl, S/N: sigma = CONT * sqrt(w dl / 0.6643) / SNR
    wl, R, snr, fw = 5000.0, 100_000, 100.0, 6.0
    w = wl * fw / 299792.458
    dl = wl / (R * detect.PIX_PER_RES)
    expect = 1e3 * detect.CONT_FACTOR * np.sqrt(w * dl / 0.66428) / snr
    assert detect.sigma_ew(wl, R, snr, fw) == pytest.approx(expect, rel=1e-3)


def test_isotope_mass_scaling():
    m1 = hosts.isotope_mass(0.0, 99.0, 0.02)
    assert hosts.isotope_mass(1.0, 99.0, 0.02) == pytest.approx(10 * m1)
    assert hosts.isotope_mass(0.0, 99.0, 0.04) == pytest.approx(2 * m1)


def test_validation_offset_small():
    path = ROOT / "research" / "validation_sun.csv"
    if not path.exists():
        pytest.skip("run scripts/validate_sun.py first")
    rows = list(csv.DictReader(path.open()))
    d = np.log10([float(r["ew_atlas_mA"]) / float(r["ew_model_mA"]) for r in rows])
    assert len(rows) > 50
    assert abs(d.mean()) < 0.15
    assert d.std() < 0.15
