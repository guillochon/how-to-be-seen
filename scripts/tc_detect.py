"""How much Tc an observer could detect in the archival spectra, using only Tc lines on clean continuum.

The archival limits (scripts/tc_limits.py) exclude Tc: a line cannot be deeper than the observed flux. A detection
must instead show more absorption at the Tc positions than the star itself gives there. This script injects
photospheric Tc into each archival stack and finds the abundance that an observer would detect at z = 5:

1. Clean lines. A Tc line is clean if, in the stack without Tc, the flux stays above CLEAN of the local continuum
   within +/- FWHM/2 of its centre. The continuum is the highest flux within +/- WIN/2, as for the limits.
   An observer would know which lines are clean from a model or an atlas of the star; we read it from the stack.
2. Amplitude. At each clean line, a = sum t (1 - r / cont) / sum t^2 over +/- FWHM, with t the line's predicted
   depth profile (scripts/tc_limits.py Profiles, the star's model and blanketing) at the reference abundance
   LE_REF.
3. Noise. a is measured the same way at N_NULL random positions within +/- NULL_HW of the line, in the same
   stack, that are equally clean and lie more than 2 FWHM from any Tc line. Their median and scatter are the bias and noise of each line; the scatter
   includes photon noise and the weak stellar lines that remain on clean continuum.
4. Detection. With A_i(le) the amplitude that Tc at abundance le gives line i, the optimal statistic is
   z = sum w_i (a_i - mu_i) / sqrt(sum w_i^2 s_i^2), w_i = A_i / s_i^2. The injected stack is r times the profiles
   of all Tc lines, clean or not. Weak blends at the Tc positions give the stack without Tc a z0 of its own, which
   an observer must model away; we bisect for the le that raises z by 5 above z0, and report z0.

For comparison, z_photon uses every Tc line and photon noise alone (the ideal of the paper's thresholds).

    uv run python scripts/tc_detect.py [star ...]     # writes research/tc_limits/detect.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

from beacon import atoms  # noqa: E402

CKMS = 2.998e5
CLEAN = 0.9
N_NULL = 400
NULL_HW = 20.0  # A
LE_REF = 2.0
Z = 5.0
SKIP = {"iue"}  # low resolution: almost no line is clean
RNG = np.random.default_rng(1)


def window(w, a, b):
    """Slice of the sorted wavelengths w with a <= w <= b."""
    return slice(np.searchsorted(w, a, "left"), np.searchsorted(w, b, "right"))


def amplitude(w, r, cont, wl, tv, tdep):
    """Template-weighted depth at wl; tv, tdep: template velocities and depths."""
    k = window(w, wl * (1 - tv[-1] / CKMS), wl * (1 + tv[-1] / CKMS))
    if k.stop - k.start < 3:
        return None
    t = np.interp((w[k] / wl - 1) * CKMS, tv, tdep)
    return float(np.sum(t * (1 - r[k] / cont)) / np.sum(t * t))


def local_cont(w, r, wl):
    return r[window(w, wl - T.WIN / 2, wl + T.WIN / 2)].max()


def is_clean(w, r, wl, fwhm):
    k = window(w, wl * (1 - fwhm / 2 / CKMS), wl * (1 + fwhm / 2 / CKMS))
    if k.stop - k.start < 2:
        return False
    return bool(r[k].min() / local_cont(w, r, wl) >= CLEAN)


def stack_case(key, inst, w, r, sig, lines, prof, fwhm, r_ref=None):
    """r_ref: the spectrum from which the clean lines and null positions are chosen (default r), so that a
    noisier copy of a stack can be tested with the clean lines that an observer knows from the original."""
    r_ref = r if r_ref is None else r_ref
    tc_wl = np.array([ln.wl for ln in lines])
    lo, hi = w[0] + T.WIN, w[-1] - T.WIN
    lines = [ln for ln in lines if lo < ln.wl < hi]
    if not lines:
        return None
    clean = [ln for ln in lines if is_clean(w, r_ref, ln.wl, fwhm)]
    hv = 1.0 * fwhm

    def null_positions(wl):
        """Random clean positions within +/- NULL_HW of wl, away from every Tc line: line density and the
        quality of the normalization change with wavelength, so the null must be local."""
        cand = RNG.uniform(max(lo, wl - NULL_HW), min(hi, wl + NULL_HW), 6 * N_NULL)
        far = np.min(np.abs(cand[:, None] - tc_wl[None, :]) / tc_wl[None, :] * CKMS, axis=1) > 2 * fwhm
        return [x for x in cand[far] if is_clean(w, r_ref, x, fwhm)][:N_NULL]

    def tmpl(ln, le):
        p = prof[ln.wl]
        sel = np.abs(p.v) <= hv
        return p.v[sel], 1 - p(le)[sel]

    stats = {}
    for ln in clean:
        tv, td = tmpl(ln, LE_REF)
        if td.max() <= 0:
            continue
        a_null = [amplitude(w, r, local_cont(w, r, x), x, tv, td) for x in null_positions(ln.wl)]
        a_null = np.array([a for a in a_null if a is not None])
        if len(a_null) < 30:
            continue
        stats[ln.wl] = (tv, td, float(np.median(a_null)), float(np.std(a_null)))

    def inject(le):
        rr = r.copy()
        for ln in lines:
            p = prof[ln.wl]
            k = window(w, ln.wl * (1 - p.v[-1] / CKMS), ln.wl * (1 + p.v[-1] / CKMS))
            rr[k] *= np.interp((w[k] / ln.wl - 1) * CKMS, p.v, p(le))
        return rr

    def z_clean(le, rr=None):
        rr = inject(le) if rr is None else rr
        num = den = 0.0
        for ln in clean:
            if ln.wl not in stats:
                continue
            tv, td, mu, s = stats[ln.wl]
            a = amplitude(w, rr, local_cont(w, rr, ln.wl), ln.wl, tv, td)
            p = prof[ln.wl]
            A = float(np.sum(td * (1 - np.interp(tv, p.v, p(le)))) / np.sum(td * td))
            wi = A / s**2
            num += wi * (a - mu)
            den += wi**2 * s**2
        return num / np.sqrt(den) if den > 0 else 0.0

    def z_photon(le):
        z2 = 0.0
        for ln in lines:
            p = prof[ln.wl]
            k = window(w, ln.wl * (1 - hv / CKMS), ln.wl * (1 + hv / CKMS))
            if k.stop - k.start < 2:
                continue
            d = 1 - np.interp((w[k] / ln.wl - 1) * CKMS, p.v, p(le))
            z2 += np.sum((d / sig[k]) ** 2)
        return float(np.sqrt(z2))

    # the stack without Tc: weak blends at the Tc positions bias z away from zero, so a detection is a rise of Z above it
    z0 = z_clean(LE_REF, r) if stats else 0.0

    def solve(f):
        # scan upward: at very high abundance the injected lines lower the local continuum and z can fall again
        grid = np.arange(-3.0, 8.01, 0.5)
        hit = next((g for g in grid if f(g) >= Z), None)
        if hit is None:
            return None
        if hit == grid[0]:
            return float(grid[0])
        a, b = hit - 0.5, hit
        for _ in range(8):
            m = 0.5 * (a + b)
            a, b = (m, b) if f(m) < Z else (a, m)
        return round(0.5 * (a + b), 2)

    if stats and "--debug" in sys.argv:
        print("   z_clean at le = 0, 2, 4, 6, 8:", [round(z_clean(x), 1) for x in (0, 2, 4, 6, 8)])
    return dict(n_lines=len(lines), n_clean=len(stats), z0=round(z0, 2), le_photon=solve(z_photon),
                le_clean=solve(lambda le: z_clean(le) - z0) if stats else None)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keys = args or sorted(p.stem for p in T.OUT.glob("*.json") if p.stem in T.STARS and p.stem not in T.EXCLUDED)
    out_path = T.OUT / "detect.csv"
    done = {}
    if out_path.exists() and not args:
        with out_path.open() as f:
            done = {(r["key"], r["instrument"]): r for r in csv.DictReader(f)}
    rows = list(done.values())
    for key in keys:
        npz = T.OUT / key / "stacks.npz"
        if not npz.exists():
            continue
        b = json.loads((T.OUT / f"{key}.json").read_text())
        if "tc_lines" not in b or "best" not in b:
            continue
        cal = b["best"].get("calibrated") or {}
        blanket = cal.get("blanket", 1.0)
        star = T.STARS[key]
        syn = T.synths(star)[blanket]
        d = np.load(npz)
        for inst in sorted({ln["instrument"] for ln in b["tc_lines"]} - SKIP):
            if "--inst" in " ".join(sys.argv) and f"--inst={inst}" not in sys.argv:
                continue
            if (key, inst) in done or inst + "_w" not in d.files:
                continue
            w, r, sig = d[inst + "_w"], d[inst + "_r"], d[inst + "_sig"]
            ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
            w, r, sig = w[ok], r[ok], sig[ok]
            recs = [ln for ln in b["tc_lines"] if ln["instrument"] == inst]
            fwhm = recs[0]["fwhm_kms"]
            lines = [atoms.Line(ln["wl"], ln["loggf"], ln["elow"], 43, 0 if ln["species"] == "Tc I" else 1) for ln in recs]
            prof = {ln.wl: T.Profiles(syn, ln, fwhm) for ln in lines}
            res = stack_case(key, inst, w, r, sig, lines, prof, fwhm)
            if res is None:
                continue
            row = dict(key=key, name=star.name, cp=star.cp, teff=star.teff, instrument=inst,
                       limit=cal.get("log_eps") if cal.get("instrument") == inst else "", **res)
            rows.append(row)
            print(f"{star.name[:24]:24s} {inst:10s} lines {res['n_lines']:3d} clean {res['n_clean']:3d}  "
                  f"z0 {res['z0']:+.1f}  photon {res['le_photon']}  clean-line detection {res['le_clean']}  limit {row['limit']}", flush=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)


if __name__ == "__main__":
    main()
