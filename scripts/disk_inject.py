"""Inject the absorption of a magnetospheric Tc II disk (scripts/disk_beacon.py) into archival ultraviolet spectra of
Ap stars, and find the disk that a matched filter detects against their real spectra.

Disk. A thin ring of Tc II in the magnetic equatorial plane from R to k R, seen at tilt beta. The front half covers
the sky points (x, y') with y' > 0 and 1 <= x^2 + (y' / sin beta)^2 <= k^2 (units of R). It corotates rigidly
with the star, so a point at sky position x moves at v sin i * x along the line of sight, with the rotation
axis projected along y'. The column along a sightline is Sigma / sin beta. Each line has a Gaussian profile of
Doppler width B_KMS (thermal Tc at 1e4 K plus a little turbulence) and line-centre optical depth
tau0 = (pi e^2 / m_e c) f lambda N_l / (sqrt(pi) b), with the lower levels in Boltzmann equilibrium at T_EXC.
The absorbed fraction of the disc-integrated flux is the mean of (1 - exp(-tau)) over the stellar disc (no limb
darkening), and it is convolved with a Gaussian of the instrument's resolution.

Detection. The rest-frame stack r (scripts/tc_limits.py) is multiplied by 1 - A. The statistic is the
template-weighted absorption, s(v) = sum t_i (1 - r_i(v)) / sigma_i^2, with the template shifted by trial
velocities v. As in scripts/tc_matched_filter.py, z compares s(0) with the median and scatter of s at
|v| > 3 v_line, where the Tc lines fall on random places of the stellar spectrum, so the scatter includes
blending with the star's own lines. z_ideal is the same template against photon noise alone. We find the
surface density Sigma that gives z = 5 for each ring size k and tilt, and the inventory Sigma m pi R^2 (k^2 - 1).

v sin i comes from the line widths of each star's optical stack: sqrt(FWHM^2 - FWHM_instrument^2) / 1.73.

    uv run python scripts/disk_inject.py     # writes research/disk_inject.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

from beacon import ROOT, atoms, comet, hosts  # noqa: E402
from beacon.const import AMU, LINE_CROSS, RSUN  # noqa: E402

C_KMS = 2.998e5
B_KMS = 1.5
T_EXC = 3000.0
R_INST = {"e230h": 114000, "ghrs_echb": 80000, "ghrs_g270m": 25000}
R_OPT = {"espadons": 65000, "narval": 65000, "uves": 80000}
# star, ultraviolet stack: three Ap stars and two non-magnetic stars for comparison
CASES = [("hd9996", "e230h"), ("hd112413", "ghrs_g270m"),
         ("hd48915", "e230h"), ("hd175640", "e230h")]
KS = (1.1, 1.5, 2.0, 3.0, 5.0)
BETAS = (10.0, 30.0)
NGRID = 301
Z_DETECT = 5.0


def vsini(b: dict) -> float:
    for inst, r in R_OPT.items():
        w = [ln["fwhm_kms"] for ln in b["tc_lines"] if ln["instrument"] == inst]
        if w:
            return float(np.sqrt(max(w[0] ** 2 - (C_KMS / r) ** 2, 0.0)) / 1.73)
    raise ValueError("no optical stack")


class Disk:
    def __init__(self, beta_deg, k, vsini_kms, lines, U):
        g = np.linspace(-1, 1, NGRID)
        x, y = np.meshgrid(g, g)
        on = x**2 + y**2 <= 1
        s = np.sin(np.radians(beta_deg))
        q = x**2 + (y / s) ** 2
        cov = on & (y > 0) & (q >= 1) & (q <= k * k)
        self.v = vsini_kms * x[cov]  # line-of-sight velocity of the covered points
        self.n_on = on.sum()
        self.s = s
        self.lines = lines
        self.U = U

    def absorption(self, wl, sigma_col):
        """Absorbed fraction of the flux on wavelengths wl for surface density sigma_col (ions cm^-2)."""
        n_los = sigma_col / self.s
        A = np.zeros_like(wl)
        if not len(self.v):
            return A
        vb = np.linspace(self.v.min(), self.v.max(), 41) if np.ptp(self.v) > 0.2 * B_KMS else np.array([self.v.mean()])
        wv = np.histogram(self.v, bins=len(vb), range=(vb[0] - 1e-6, vb[-1] + 1e-6))[0] / self.n_on if len(vb) > 1 \
            else np.array([len(self.v) / self.n_on])
        for ln in self.lines:
            if not (wl[0] < ln.wl < wl[-1]):
                continue
            nl = n_los * np.exp(-atoms.HC_K * ln.elow / T_EXC) / self.U
            tau0 = LINE_CROSS * 10**ln.loggf * nl * ln.wl * 1e-8 / (np.sqrt(np.pi) * B_KMS * 1e5)
            sel = np.abs(wl - ln.wl) < ln.wl * (np.abs(self.v).max() + 6 * B_KMS) / C_KMS + 0.05
            dv = (wl[sel] / ln.wl - 1) * C_KMS
            for vv, ww in zip(vb, wv, strict=True):
                A[sel] += ww * -np.expm1(-tau0 * np.exp(-(((dv - vv) / B_KMS) ** 2)))
        return np.minimum(A, 1.0)


def statistic(w, r, sig, t, vlim):
    """z of the template t at v = 0 against shifted copies, and z against photon noise alone."""
    sel = t > 1e-4 * t.max()
    wt = t[sel] / sig[sel] ** 2
    vg = np.arange(-300.0, 300.01, 1.0)
    s = np.array([np.sum(wt * (1 - np.interp(w[sel] * (1 + v / C_KMS), w, r))) for v in vg])
    off = s[np.abs(vg) > vlim]
    z = (s[np.argmin(np.abs(vg))] - np.median(off)) / np.std(off)
    z_ideal = np.sum(wt * t[sel]) / np.sqrt(np.sum(t[sel] ** 2 / sig[sel] ** 2))
    return float(z), float(z_ideal)


def photospheric(key, w, r, sig, lines, fwhm, vlim):
    """The same test for Tc II spread through the star's atmosphere: the log eps at which z = 5, and the
    z that photon noise alone would give there."""
    syn = T.synths(T.STARS[key])[1.0]
    profs = [(ln, T.Profiles(syn, ln, fwhm)) for ln in lines]

    def absorb(le):
        rr = np.ones_like(w)
        for ln, pr in profs:
            v = (w / ln.wl - 1) * C_KMS
            sel = np.abs(v) < pr.v[-1]
            rr[sel] *= np.interp(v[sel], pr.v, pr(le))
        return 1 - rr

    lo, hi = -3.0, 7.0
    if statistic(w, r * (1 - absorb(hi)), sig, absorb(hi), vlim)[0] < Z_DETECT:
        return dict(detected=False, z_at_max=statistic(w, r * (1 - absorb(hi)), sig, absorb(hi), vlim)[0])
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        A = absorb(mid)
        lo, hi = (mid, hi) if statistic(w, r * (1 - A), sig, A, vlim)[0] < Z_DETECT else (lo, mid)
    le = 0.5 * (lo + hi)
    A = absorb(le)
    # photon-noise threshold for comparison
    lo2, hi2 = -3.0, 7.0
    for _ in range(16):
        mid = 0.5 * (lo2 + hi2)
        A2 = absorb(mid)
        lo2, hi2 = (mid, hi2) if statistic(w, r * (1 - A2), sig, A2, vlim)[1] < Z_DETECT else (lo2, mid)
    return dict(detected=True, log_eps_blind=le, log_eps_photon=0.5 * (lo2 + hi2))


def main():
    sp = comet.species()["Tc II"]
    U = float(atoms.partition(43, 1, np.array([T_EXC]))[0])
    out = {}
    for key, inst in CASES:
        b = json.loads((T.OUT / f"{key}.json").read_text())
        d = np.load(T.OUT / key / "stacks.npz")
        w, r, sig = d[inst + "_w"], d[inst + "_r"], d[inst + "_sig"]
        ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
        w, r, sig = w[ok], r[ok], sig[ok]
        # lines with data on both sides (the GHRS stacks have gaps between their windows)
        dw = np.median(np.diff(w))
        lines = [ln for ln in sp.lines if w[0] + 1 < ln.wl < w[-1] - 1
                 and np.sum(np.abs(w - ln.wl) < 0.5) > 0.8 / dw]
        vs = vsini(b)
        star = T.STARS[key]
        R = star.radius * RSUN
        dpix = np.median(np.diff(w)) / np.median(w) * C_KMS
        lsf = (C_KMS / R_INST[inst]) / 2.355 / dpix
        z0, _ = statistic(w, r, sig, Disk(30.0, 1.1, vs, lines, U).absorption(w, 1e13), max(3 * vs, 15.0))
        fwhm = [ln["fwhm_kms"] for ln in b["tc_lines"] if ln["instrument"] == inst][0]
        phot = photospheric(key, w, r, sig, [ln for ln in lines], fwhm, max(3 * vs, 15.0))
        print(f"  photospheric Tc II: {phot}")
        rec = dict(photospheric=phot, name=star.name, cp=star.cp, teff=star.teff, radius_rsun=star.radius, vsini=vs, instrument=inst,
                   n_lines=len(lines), snr=float(np.median(1 / sig)), z_without_tc=z0, designs={})
        for beta in BETAS:
            for k in KS:
                disk = Disk(beta, k, vs, lines, U)

                def z_at(log_sig, disk=disk):
                    A = gaussian_filter1d(disk.absorption(w, 10**log_sig), lsf)
                    t = gaussian_filter1d(disk.absorption(w, 10**log_sig), lsf)
                    return statistic(w, r * (1 - A), sig, t, max(3 * vs, 15.0))

                lo, hi = 9.0, 20.0
                z_sat = z_at(hi)[0]
                if z_sat < Z_DETECT:
                    rec["designs"][f"{beta:g}|{k:g}"] = dict(z_saturated=z_sat, detected=False)
                    continue
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if z_at(mid)[0] < Z_DETECT else (lo, mid)
                ls = 0.5 * (lo + hi)
                z, zi = z_at(ls)
                m = 10**ls * sp.mass * AMU * np.pi * R**2 * (k * k - 1)
                p = hosts.upkeep_power(m * hosts.NUCLIDES["99Tc"]["A"] / sp.mass, "99Tc capture") / 1e7
                rec["designs"][f"{beta:g}|{k:g}"] = dict(detected=True, z_saturated=z_sat, log_sigma=ls, z=z, z_ideal=zi, mass_kg=m / 1e3,
                                                          sagan_K=(np.log10(p) - 6) / 10)
        best = min((v for v in rec["designs"].values() if v["detected"]), key=lambda v: v["mass_kg"], default=None)
        rec["best"] = best
        out[key] = rec
        print(f"{star.name[:22]:22s} {star.cp:6s} {inst:10s} vsini {vs:5.1f}  S/N {rec['snr']:.0f}  lines {len(lines)}  "
              f"z(no Tc) {z0:+.1f}")
        for kk, v in rec["designs"].items():
            if not v["detected"]:
                print(f"   beta|k {kk:8s} not detected: z = {v['z_saturated']:.1f} with saturated lines")
            else:
                print(f"   beta|k {kk:8s} log Sigma {v['log_sigma']:.2f}  z_ideal at threshold {v['z_ideal']:.1f}  "
                      f"{v['mass_kg']:.1e} kg  K {v['sagan_K']:.2f}")
    (ROOT / "research" / "disk_inject.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
