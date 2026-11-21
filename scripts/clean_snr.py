"""The penalty of the clean-line receiver at lower signal-to-noise ratio (referee comment 2.7).

In HD 175640 at its archival S/N, a search with only the Tc lines on clean continuum needs 1.6 dex more Tc than the
ideal receiver (scripts/tc_detect.py). At the S/N of the human-budget reach (about 30 per pixel), photon noise is a
larger share of the noise at the clean lines, so the penalty may shrink. We add Gaussian noise to the STIS E230H
stack to bring its median S/N per pixel to each target, keep the clean lines and null positions of the original
stack (an observer would know them from a model or a high-S/N spectrum), and repeat the test N_REAL times.

    uv run python scripts/clean_snr.py      # writes research/clean_snr.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_detect as D  # noqa: E402
import tc_limits as T  # noqa: E402

from beacon import ROOT, atoms  # noqa: E402

KEY, INST = "hd175640", "e230h"
TARGETS = (None, 100.0, 31.0)
N_REAL = 3


def main():
    b = json.loads((T.OUT / f"{KEY}.json").read_text())
    cal = b["best"].get("calibrated") or {}
    syn = T.synths(T.STARS[KEY])[cal.get("blanket", 1.0)]
    d = np.load(T.OUT / KEY / "stacks.npz")
    w, r, sig = d[INST + "_w"], d[INST + "_r"], d[INST + "_sig"]
    ok = np.isfinite(r) & np.isfinite(sig) & (sig > 0)
    w, r, sig = w[ok], r[ok], sig[ok]
    recs = [ln for ln in b["tc_lines"] if ln["instrument"] == INST]
    fwhm = recs[0]["fwhm_kms"]
    lines = [atoms.Line(ln["wl"], ln["loggf"], ln["elow"], 43, 0 if ln["species"] == "Tc I" else 1) for ln in recs]
    prof = {ln.wl: T.Profiles(syn, ln, fwhm) for ln in lines}
    snr0 = float(np.median(r / sig))
    out = dict(star=KEY, instrument=INST, snr_original=snr0, cases=[])
    rng = np.random.default_rng(7)
    for tgt in TARGETS:
        reals = []
        for _ in range(1 if tgt is None else N_REAL):
            if tgt is None:
                rr, ss = r, sig
            else:
                extra = np.sqrt(np.maximum((r / tgt) ** 2 - sig**2, 0.0))
                rr, ss = r + rng.normal(0.0, 1.0, len(r)) * extra, np.sqrt(sig**2 + extra**2)
            res = D.stack_case(KEY, INST, w, rr, ss, lines, prof, fwhm, r_ref=r)
            reals.append(res)
            print(f"S/N {tgt or snr0:6.1f}: photon {res['le_photon']}, clean {res['le_clean']}, z0 {res['z0']}", flush=True)
        lp = [x["le_photon"] for x in reals if x["le_photon"] is not None]
        lc = [x["le_clean"] for x in reals if x["le_clean"] is not None]
        out["cases"].append(dict(snr=tgt or snr0, le_photon=float(np.mean(lp)), le_clean=float(np.mean(lc)),
                                 penalty_dex=float(np.mean(lc) - np.mean(lp)), realizations=reals))
    (ROOT / "research" / "clean_snr.json").write_text(json.dumps(out, indent=1))
    for c in out["cases"]:
        print(f"S/N {c['snr']:6.1f}: clean-line penalty {c['penalty_dex']:.2f} dex (x{10 ** c['penalty_dex']:.0f} in inventory)")


if __name__ == "__main__":
    main()
