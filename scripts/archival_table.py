"""Tabulate the archival Tc limits against the paper's beacon thresholds, host class by host class.

Reads research/tc_limits/summary.csv (scripts/tc_batch.py --summary), the thresholds of research/results.json
(scripts/results.py) and research/quiet_host.json (scripts/quiet_host.py). Each star's K assumes that the Tc fills
the mixed mass of its host class (scripts/tc_limits.py, host_class). It is compared with the threshold of the same
class at S/N = 300: the quiet late-B beacon without a wind, the Am star read in Tc II by the 2.4 m space echelle
(the A0 star for the hottest A stars), and the Sun or the F dwarf read in Tc I by an 8.2 m telescope.

Writes research/tc_limits/k_limits.md (every star), paper/tables/archival.tex (the lowest K of each class),
paper/data/archival_limits.csv (the machine-readable table of every star, its parameters and their sources) and
research/tc_limits/archival_stats.json (the numbers quoted in the paper).

    uv run python scripts/archival_table.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tc_limits as T  # noqa: E402

GROUND = "8.2 m, R = 140,000|300"
SPACE = "2.4 m space UV, R = 114,000|300"
CLASSES = ("quiet late-B", "A star", "HB", "FGK")
TOP = 5  # stars per class in the paper's table
HUMANITY_K = 0.72
# Types of the hand-checked stars (research/tc_candidates.md) and the labels the paper uses.
HAND_KIND = {**{k: "HgMn" for k in T.QUIET_HAND}, "hd48915": "AmFm", "hd72660": "AmFm", "hd209625": "AmFm",
             "hd189849": "AmFm", "hd214994": "AmFm", "hd27962": "AmFm", "hd9996": "ApBp", "hd101065": "ApBp",
             "hd217522": "ApBp", "hd965": "ApBp", "cpd-62_2717": "ApBp"}
LABEL = {"ApBp": "Ap/Bp", "AmFm": "Am", "HgMn": "HgMn", "HB": "BHB", "metal-poor": "metal-poor",
         "normal": "normal", "uncertain": "uncertain"}


def thresholds() -> dict:
    res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    quiet = json.loads((ROOT / "research" / "quiet_host.json").read_text())
    return {
        "quiet late-B": dict(K=quiet["cases"]["no wind"]["sagan_K"], K_wind=quiet["cases"]["wind"]["sagan_K"],
                             log_eps=quiet["log_eps"], teff=quiet["teff"]),
        "A star": dict(K=res["Am"]["species"]["Tc II"][SPACE]["sagan_K_best"],
                       K_hot=res["A0"]["species"]["Tc II"][SPACE]["sagan_K_best"],
                       log_eps=res["Am"]["species"]["Tc II"][SPACE]["log_eps"], teff=res["Am"]["teff"]),
        "HB": dict(K=res["Am"]["species"]["Tc II"][SPACE]["sagan_K_best"],
                   log_eps=res["Am"]["species"]["Tc II"][SPACE]["log_eps"], teff=res["Am"]["teff"]),
        "FGK": dict(K=res["G"]["species"]["Tc I"][GROUND]["sagan_K_best"],
                    K_F=res["F"]["species"]["Tc I"][GROUND]["sagan_K_best"],
                    log_eps=res["G"]["species"]["Tc I"][GROUND]["log_eps"], teff=res["G"]["teff"]),
    }


def kind(r: dict) -> str:
    """The star's type as the compilation, the list or research/tc_candidates.md gives it."""
    if r["key"] in HAND_KIND:
        return HAND_KIND[r["key"]]
    s = r["source"]
    u = re.search(r"research/(\w+)\.md, ([\w-]+)\)", s)
    m = re.search(r"compilation \((\w+)\)", s)
    if u:
        return {"normal": "normal"}.get(u.group(2), u.group(2))
    return m.group(1) if m else "hand-checked"


def rows() -> list[dict]:
    out = []
    for r in csv.DictReader((T.OUT / "summary.csv").open(encoding="utf-8")):
        if not r["log_eps"] or not r["K_host"]:
            continue
        st = T.STARS[r["key"]]
        out.append(dict(key=r["key"], name=r["name"], teff=float(r["teff"]), kind=kind(r), host=r["host"],
                        log_eps=float(r["log_eps"]), blanket="uncal." if st.fe is None else f"x{float(r['blanket']):g}",
                        line=r["line"], instrument=r["instrument"], K_atm=float(r["K_atmosphere"]),
                        K_host=float(r["K_host"])))
    return out


def latex_name(name: str) -> str:
    """The star's name for the paper: the Bayer or common name if it has one, else the HD number."""
    name = re.sub(r"^(.+?) \((HD|BD|CD)[^)]*\)$", r"\g<1>", name)  # "53 Tau (HD 27295)" -> "53 Tau"
    name = name.replace("chi Lup A", r"$\chi$~Lup~A").replace("mu Lep", r"$\mu$~Lep").replace("kappa Cnc", r"$\kappa$~Cnc")
    name = name.replace("o Peg", r"$o$~Peg").replace("eta Lep", r"$\eta$~Lep").replace("HD ", "HD~").replace("BD-", "BD$-$").replace("CD-", "CD$-$")
    return name


INSTRUMENTS = {"uves": "UVES", "harps": "HARPS", "espresso": "ESPRESSO", "feros": "FEROS", "espadons": "ESPaDOnS",
               "narval": "Narval", "elodie": "ELODIE", "sophie_hr": "SOPHIE", "sophie_he": "SOPHIE", "iue": "IUE",
               "e230h": "STIS E230H", "e230m": "STIS E230M", "ghrs_g270m": "GHRS G270M", "ghrs_echb": "GHRS Ech-B"}


def main():
    th = thresholds()
    rs = rows()
    # Delta K against the star's own threshold, in its own model (scripts/own_thresholds.py); the mixed mass cancels
    own = {r["key"]: r for r in csv.DictReader((T.OUT / "own_thresholds.csv").open(encoding="utf-8"))}
    for r in rs:
        o = own[r["key"]]
        r["thr_own"] = float(o["threshold_own"])
        r["dK"] = (r["log_eps"] - r["thr_own"]) / 10
        r["mf3"] = float(o["limit_mf3"]) if o["limit_mf3"] not in ("", "nan") else None

    # every star, for research/
    lines = ["# Archival Tc limits and K, all stars", "",
             "Generated by `scripts/archival_table.py` from `research/tc_limits/summary.csv`. log ε(Tc) = log(N_Tc/N_H) + 12.",
             "K is the Sagan type of the upkeep of the largest Tc inventory that the spectrum allows (Tc bred from 98Mo).",
             "K, host assumes that the Tc fills the mixed mass of the star's host class in the paper: the atmosphere of a quiet",
             "late-B star (HgMn and normal late-B stars), the turbulent layer of an A star (log ΔM/M* = -5.5), or the",
             "convection zone of an FGK star (Teff ≤ 6400 K). K, atmosphere assumes that it stays above τ5000 = 3 in every star.",
             "ΔK = (log ε(Tc) limit - log ε(Tc) threshold) / 10, with the threshold of the class's beacon (Tc I from 8.2 m",
             "at or below 6400 K, Tc II from the 2.4 m space echelle above) computed in the star's own model at S/N = 300",
             "(scripts/own_thresholds.py); the mixed mass cancels. MF 3σ: the upper limit in the clean-line matched filter.",
             "Opacity: the blanketing factor set by the star's Fe lines, or uncal. if it has no published Fe.", "",
             "| # | Star | Type | Host class | Teff (K) | log ε(Tc) ≤ | Threshold | MF 3σ | Opacity | K, host | ΔK | K, atmosphere | Instrument |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(rs, key=lambda r: r["dK"]), 1):
        mf = f"{r['mf3']:.1f}" if r["mf3"] is not None else ""
        lines.append(f"| {i} | {r['name']} | {r['kind']} | {r['host']} | {r['teff']:.0f} | {r['log_eps']:.1f} | "
                     f"{r['thr_own']:.2f} | {mf} | {r['blanket']} | "
                     f"{r['K_host']:.2f} | {r['dK']:+.2f} | {r['K_atm']:.2f} | {r['instrument']} |")
    (T.OUT / "k_limits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the paper's table: the lowest K of each class
    tex = [r"\begin{tabular}{llrrrlrr}", r"\toprule",
           r"Star & Type & $T_\mathrm{eff}$ (K) & $\log\epsilon_\mathrm{Tc}$ & Threshold & Line, instrument & $K$ & $\Delta K$ \\",
           r"\midrule"]
    labels = {"quiet late-B": r"Quiet late-B hosts (atmosphere)", "A star": r"Am and Ap stars (turbulent layer)",
              "HB": r"Horizontal-branch stars (turbulent layer, uncalibrated)", "FGK": r"FGK stars (convection zone)"}
    for c in CLASSES:
        tex.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{labels[c]}}}}} \\")
        for r in sorted((r for r in rs if r["host"] == c), key=lambda r: r["K_host"])[:TOP]:
            wl = r["line"].split()[-1]
            line = rf"Tc\,\textsc{{{r['line'].split()[1].lower()}}} {float(wl):.0f}, {INSTRUMENTS.get(r['instrument'], r['instrument'])}"
            tex.append(rf"{latex_name(r['name'])} & {LABEL[r['kind']]} & {r['teff']:.0f} & $\leq {r['log_eps']:+.1f}$ & "
                       rf"${r['thr_own']:+.1f}$ & {line} & "
                       rf"{r['K_host']:.2f} & ${r['dK']:+.2f}$ \\")
        if c != CLASSES[-1]:
            tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "archival.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # the machine-readable table that accompanies the paper
    sample = {"".join(r["star"].lower().split()): r for r in csv.DictReader((T.OUT / "cp_sample.csv").open(encoding="utf-8"))}
    data = []
    for r in sorted(rs, key=lambda r: (CLASSES.index(r["host"]), r["K_host"])):
        st, c = T.STARS[r["key"]], sample.get(r["key"], {})
        b = json.loads((T.OUT / f"{r['key']}.json").read_text())
        if r["key"] in HAND_KIND:
            params = fe_src = st.source.split("; R ")[0]
        elif c.get("list"):
            params, fe_src = c.get("params_ref", ""), c.get("fe_ref", "")
        else:
            params, fe_src = "Ghazaryan et al. (2018) compilation, 2018MNRAS.480.2953G", c.get("fe_ref", "")
        rsrc = ("V, parallax and Teff (scripts/radii.py)" if T.RADII_SOURCE in st.source else
                "Gaia DR3 FLAME" if "FLAME" in st.source else "published (see parameter source)")
        data.append(dict(star=r["name"], simbad=c.get("simbad", "").strip(), ra_deg=round(st.ra, 5), dec_deg=round(st.dec, 5),
                         type=LABEL[r["kind"]], host_class=r["host"], teff_K=st.teff, logg=st.logg, vsini_kms=c.get("vsini", ""),
                         log_eps_fe=round(sum(st.fe) / 2, 2) if st.fe else "", parameter_source=params, fe_source=fe_src,
                         radius_rsun=round(st.radius, 3), radius_source=rsrc, model_grid=T.grid_name(st),
                         opacity_factor=r["blanket"], log_eps_tc_max=r["log_eps"], line=r["line"],
                         instrument=INSTRUMENTS.get(r["instrument"], r["instrument"]),
                         spectra=" ".join(f"{INSTRUMENTS.get(i, i)}:{v['n']}" for i, v in b["spectra"].items()),
                         K_max=round(r["K_host"], 3), log_eps_tc_threshold_own=r["thr_own"], delta_K=round(r["dK"], 3),
                         log_eps_tc_max_matched_filter=r["mf3"] if r["mf3"] is not None else "",
                         K_max_atmosphere=round(r["K_atm"], 3)))
    out = ROOT / "paper" / "data" / "archival_limits.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

    # numbers for the text
    stats = dict(n_stars=len(rs), by_class=dict(Counter(r["host"] for r in rs)),
                 by_kind=dict(Counter(r["kind"] for r in rs)),
                 at_or_below_threshold={c: [r["name"] for r in rs if r["host"] == c and r["dK"] <= 0] for c in CLASSES},
                 below_humanity=[r["name"] for r in rs if r["K_host"] < HUMANITY_K],
                 best={c: min((dict(name=r["name"], log_eps=r["log_eps"], K=r["K_host"], dK=round(r["dK"], 3))
                               for r in rs if r["host"] == c), key=lambda d: d["K"]) for c in CLASSES},
                 instruments=dict(Counter(INSTRUMENTS.get(r["instrument"], r["instrument"]) for r in rs).most_common()),
                 thresholds=th)
    (T.OUT / "archival_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "thresholds"}, indent=1))


if __name__ == "__main__":
    main()
