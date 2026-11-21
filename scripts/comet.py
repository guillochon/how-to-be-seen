"""Doped exocomets: detection thresholds, column masses, activity, design and production cost.

Writes research/comet.json, paper/tables/comet.tex (thresholds) and paper/tables/comet_design.tex
(cost of a beacon that a monitoring program is likely to catch, beacon/comet.py event_rate, for four comet designs).

    uv run python scripts/comet.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, comet, detect, hosts  # noqa: E402
from beacon.const import YR  # noqa: E402

SNRS = (100.0, 300.0)
SNR = 300.0
GROUND = "3.6 m, R = 115,000"
SPACE = "2.4 m space UV, R = 114,000"
UVBLUE = "8.2 m UV-blue, R = 80,000"
ROWS = [("Tc II", (SPACE, UVBLUE)), ("Pm II", (GROUND,)), ("Ra II", (GROUND,)),
        ("Ca II", (GROUND,)), ("Cr II", (SPACE,))]
TEL = {GROUND: "3.6 m", SPACE: "2.4 m sp.", UVBLUE: "8 m UV"}

# Comet orbit and monitoring. Refractory grains sublimate within several tens of stellar radii
# (Kiefer et al. 2019); we take a parabolic orbit with periastron 10 R* and release inside 30 R*.
Q_RSTAR = 10.0
RSUB_RSTAR = 30.0
SPECTRA_PER_YR = comet.SPECTRA_PER_YR
# Dopants for the design study: species, instrument, production route (None: uranium chain).
DOPANTS = [("Ra II", GROUND, None), ("Tc II", SPACE, "99Tc capture"), ("Pm II", GROUND, "145Pm capture")]


def radius(teff: float) -> float:
    """R/Rsun of a main-sequence star of this Teff (Pecaut & Mamajek 2013; Mamajek table)."""
    t, r = [], []
    for ln in (ROOT / "research" / "mamajek" / "EEM_dwarf_UBVIJHK_colors_Teff.txt").read_text().splitlines():
        tok = ln.split()
        if len(tok) > 8 and tok[0].endswith("V") and tok[0][0] in "BAFGKM" and tok[1].isdigit():
            try:
                t.append(float(tok[1]))
                r.append(float(tok[6]))
            except ValueError:
                continue
    o = np.argsort(t)
    return float(np.interp(teff, np.array(t)[o], np.array(r)[o]))


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def thresholds(sp, inst, out):
    r_star = radius(comet.T_EXC)
    lines_tex = [r"\begin{tabular}{lllrrr}", r"\toprule",
                 r"Ion & Lines (\AA) & Tel. & $N_{300}$ (cm$^{-2}$) & $N_{300}/N_\mathrm{Fe}$ & Mass (kg) \\",
                 r"\midrule"]
    for name, insts in ROWS:
        s = sp[name]
        for iname in insts:
            i = inst[iname]
            n = {snr: comet.threshold(s, i, snr) for snr in SNRS}
            best = comet.best_lines(s, n[300.0], i, k=2)
            m = comet.column_mass(n[300.0], s.mass, r_star)
            rec = dict(n100=n[100.0], n300=n[300.0], ratio_fe=n[300.0] / comet.N_FEII, mass_g=m,
                       best=[round(b.wl, 1) for b in best])
            out["rows"][f"{name}|{iname}"] = rec
            ref = name in ("Ca II", "Cr II")
            lab = name.replace(" II", r"\,\textsc{ii}") + (" (natural)" if ref else "")
            lines_tex.append(f"{lab} & {', '.join(f'{b:.0f}' for b in rec['best'])} & {TEL[iname]} & "
                             f"{sci(n[300.0])} & {sci(rec['ratio_fe'])} & {'--' if ref else sci(m / 1e3)} \\\\")
            print(f"{name} {iname:30s} N100={n[100.0]:.2e} N300={n[300.0]:.2e} ratio={rec['ratio_fe']:.1e} "
                  f"M={m / 1e3:.1e} kg best={rec['best']}")
    lines_tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "comet.tex").write_text("\n".join(lines_tex) + "\n")
    out["r_star_rsun"] = r_star


def designs(sp, inst, out):
    am = dict(name="Am", m=1.8, r=radius(comet.T_EXC), t=comet.T_EXC)
    scen = [
        ("beta Pic-like", am, comet.DELTA_V, None),
        ("timed release", am, comet.DELTA_V, 1.0),
        ("timed, narrow", am, 3.0, 1.0),
    ]
    labels = {"beta Pic-like": r"$\beta$~Pic-like", "timed release": "Timed release",
              "timed, narrow": r"+ $\Delta v = 3$~km~s$^{-1}$"}
    lines_tex = [r"\begin{tabular}{llrrrl}", r"\toprule",
                 r"Design & Dopant & $\eta$ & Events (yr$^{-1}$) & Per event (kg) & Cost \\", r"\midrule"]
    out["design"] = dict(q_rstar=Q_RSTAR, rsub_rstar=RSUB_RSTAR, spectra_per_yr=SPECTRA_PER_YR, rows={})
    for k, (sname, host, dv, eta_set) in enumerate(scen):
        tim = comet.event_timing(host["m"], host["r"], Q_RSTAR, RSUB_RSTAR)
        eta = tim["eta_nat"] if eta_set is None else eta_set
        first = True
        for name, iname, route in DOPANTS:
            s = sp[name]
            cadence = comet.SPECTRA_PER_YR_SPACE if inst[iname].space else comet.SPECTRA_PER_YR
            f_events = comet.event_rate(tim["t_transit_s"], cadence)
            n = comet.threshold(s, inst[iname], SNR, dv_kms=dv, t_exc=host["t"])
            m_event = comet.column_mass(n, s.mass, host["r"]) / eta
            m_yr = m_event * f_events
            rec = dict(n_thr=n, eta=eta, events_per_yr=f_events, mass_event_g=m_event, mass_per_yr_g=m_yr,
                       **{k2: v for k2, v in tim.items()})
            if route is None:
                rec["uranium_per_yr_g"] = m_yr * hosts.uranium_per_radium()
                cost = f"U: {sci(rec['uranium_per_yr_g'] / 1e3)} kg yr$^{{-1}}$"
            else:
                nuc = hosts.ROUTES[route][0]
                p = hosts.production_energy(m_yr * hosts.NUCLIDES[nuc]["A"] / s.mass, route) / YR / 1e7
                rec.update(power_W=p, sagan_K=(np.log10(p) - 6) / 10, route=route)
                cost = f"{sci(p)} W, $K = {rec['sagan_K']:.2f}$"
            out["design"]["rows"][f"{sname}|{name}"] = rec
            lab = name.replace(" II", r"\,\textsc{ii}")
            lines_tex.append(f"{labels[sname] if first else ''} & {lab} & "
                             f"{(f'{eta:.3f}' if eta < 1 else '1') if first else ''} & "
                             f"{f_events:.2g} & {sci(m_event / 1e3)} & {cost} \\\\")
            first = False
            print(f"{sname:30s} {name} eta={eta:.3f} t_tr={tim['t_transit_s'] / 3600:.1f} h "
                  f"t_act={tim['t_active_s'] / 86400:.1f} d f={f_events:.1f}/yr N={n:.1e} "
                  f"M_event={m_event / 1e3:.1e} kg " + (f"K={rec['sagan_K']:.2f}" if route else
                                                        f"U={rec['uranium_per_yr_g'] / 1e3:.1e} kg/yr"))
        if k != len(scen) - 1:
            lines_tex.append(r"\addlinespace")
    lines_tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "comet_design.tex").write_text("\n".join(lines_tex) + "\n")


def main():
    sp = comet.species()
    inst = {i.name: i for i in detect.INSTRUMENTS}
    out = dict(alpha=comet.ALPHA, delta_v=comet.DELTA_V, t_exc=comet.T_EXC, n_feii=comet.N_FEII,
               n_crii_measured=comet.CR_OVER_FE * comet.N_FEII, rows={})
    thresholds(sp, inst, out)
    designs(sp, inst, out)
    (ROOT / "research" / "comet.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"R* = {out['r_star_rsun']:.2f} Rsun; measured N(Cr II) = {out['n_crii_measured']:.2e}")


if __name__ == "__main__":
    main()
