"""Is Tc still pushed upward when a magnetic field holds the whole beacon above tau5000 = 1e-5? (referee 2.9)

A field that stops Tc from leaving the quiet late-B star at tau5000 ~ 1e-5 concentrates the column that would fill
the atmosphere into the thin layer above it, raising the local abundance by ~4 dex (research/tc_drift.json,
held_layer). Radiation supports Tc there only if its lines are not too saturated at that abundance. We compute
g_rad / g for Tc in the 12,250 K model (scripts/levitation.py method) at the held abundance of the threshold beacon
and of the beacon run on humanity's power (12 times the threshold inventory), uniform with depth, and read it from
tau5000 = 1e-6 to 1e-3.

    uv run python scripts/held_grad.py      # writes research/held_grad.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT  # noqa: E402
from beacon import levitation as L  # noqa: E402

TEFF, LOGG, VTURB = 12250, 4.0, 0.5
TAUS = (-6.0, -5.5, -5.0, -4.5, -4.0, -3.0)


def main():
    drift = json.loads((ROOT / "research" / "tc_drift.json").read_text())
    reach = json.loads((ROOT / "research" / "reach_budget.json").read_text())
    le_thr = drift["log_eps"]
    le_held = drift["held_layer"]["1e-05"]["log_eps_in_layer"]
    le_human = next(r["log_eps"] for r in reach["rows"] if r["budget"] == "humanity")
    cases = {"threshold, held": le_held, "humanity, held": le_held + (le_human - le_thr)}
    atm = L.Atmosphere(TEFF, LOGG, VTURB)
    corr = L.flux_correction(atm, TEFF, LOGG)
    nb = L.neighbour_lines(atm, exclude_z=43)
    el = L.tc_lines()
    lt = np.log10(atm.tau5)
    print(f"model top at log tau5000 = {lt[0]:.2f}")
    out = dict(teff=TEFF, logg=LOGG, taus=TAUS, cases={})
    for name, le in cases.items():
        g_ion, n = L.grad_element(atm, el, 98.906, le, nb, corr)
        _, dw = L.element_grad(atm, g_ion, 43)
        ratio = [float(np.interp(x, lt, np.log10(np.maximum(dw, 1e-30)))) - LOGG for x in TAUS]
        out["cases"][name] = dict(log_eps=le, nlines=n, log_grad_over_g=ratio)
        print(f"{name:18s} log eps {le:5.2f}: log(g_rad/g) at log tau " +
              "  ".join(f"{x:+.1f}: {v:+.2f}" for x, v in zip(TAUS, ratio, strict=True)), flush=True)
    (ROOT / "research" / "held_grad.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
