"""Search the archival stacks for Tc absorption with a matched filter, as a check that no star shows Tc.

For each star, instrument and ion (Tc I, Tc II), the absorption 1 - r of the rest-frame stack is summed at the
Tc line positions with weights gf 10**(-5040/Teff chi_low) / sigma, the predicted line strength over the noise,
set without looking at the data at the Tc positions. The sum is repeated with the lines shifted by trial
velocities v out to 300 km/s. z is the sum at v = 0 against the median and standard deviation of the sums at
|v| > FWHM, where the Tc lines fall at random places in the spectrum. A star with Tc would stand out at v = 0.

Writes research/tc_limits/matched_filter.csv and prints the highest z.

    uv run python scripts/tc_matched_filter.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

C = 2.998e5
VSPAN = 300.0  # km/s


def main():
    res = []
    for p in sorted(T.OUT.glob("*.json")):
        key = p.stem
        npz = T.OUT / key / "stacks.npz"
        if key not in T.STARS or not npz.exists() or key in T.EXCLUDED:
            continue
        b = json.loads(p.read_text())
        if "tc_lines" not in b:
            continue
        d = np.load(npz)
        teff = b["model"]["teff"]
        for inst, sp in sorted({(ln["instrument"], ln["species"]) for ln in b["tc_lines"]}):
            if inst + "_w" not in d.files:
                continue
            w, r, sig = d[inst + "_w"], d[inst + "_r"], d[inst + "_sig"]
            lines = [ln for ln in b["tc_lines"] if ln["instrument"] == inst and ln["species"] == sp]
            if len(lines) < 2:
                continue
            fwhm = lines[0]["fwhm_kms"]
            wl = np.array([ln["wl"] for ln in lines])
            strength = np.array([ln["loggf"] - 5040 / teff * ln["elow"] / 8065.54 for ln in lines])
            wt = 10.0**strength / np.interp(wl, w, sig)
            wt /= wt.sum()
            step = max(fwhm / 4, 1.0)
            vg = np.arange(-VSPAN, VSPAN + step / 2, step)
            s = np.array([np.sum(wt * (1 - np.interp(wl * (1 + v / C), w, r))) for v in vg])
            off = s[np.abs(vg) > fwhm]
            z = (s[np.argmin(np.abs(vg))] - np.median(off)) / np.std(off)
            top = sorted(zip(wt, lines, strict=True), key=lambda q: -q[0])[:3]
            res.append(dict(key=key, name=T.STARS[key].name, instrument=inst, ion=sp, n_lines=len(lines), z=round(z, 2),
                            main_lines=" ".join(f"{ln['wl']:.0f}" for _, ln in top)))
    res.sort(key=lambda q: -q["z"])
    with (T.OUT / "matched_filter.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(res[0]))
        wr.writeheader()
        wr.writerows(res)
    zs = np.array([q["z"] for q in res])
    print(f"{len(res)} stacks; median z {np.median(zs):.2f}, standard deviation {np.std(zs):.2f}; "
          f"z > 3: {int((zs > 3).sum())}, z > 4: {int((zs > 4).sum())}")
    for q in res[:10]:
        print(f"{q['z']:5.1f}  {q['name'][:24]:24s} {q['instrument']:11s} {q['ion']:6s} {q['n_lines']:3d} lines  {q['main_lines']}")


if __name__ == "__main__":
    main()
