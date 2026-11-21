"""Derived numbers quoted in the text (Sect. 5): fuel burned, delivery energy, molecular-weight
excess, and the samarium that a promethium beacon leaves behind. Writes research/derived.json.

    uv run python scripts/derived_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, hosts  # noqa: E402
from beacon.const import AMU, MEARTH, YR  # noqa: E402

KEY = "8.2 m, R = 140,000|300"
V_ORB_1AU = 29.78e5  # cm/s
MU_ION = 0.61  # mean molecular weight of fully ionized solar gas
LOG_EPS_SM_SUN = 0.96  # Asplund et al. (2009)


def main():
    res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    g = res["G"]["species"]
    out = {}
    tc = g["Tc I"][KEY]
    m_tc = tc["mass_g"]
    n_tc = m_tc / (hosts.NUCLIDES["99Tc"]["A"] * AMU)
    fissions = n_tc / hosts.NUCLIDES["99Tc"]["fy"]
    out["sun_tc_fissions"] = fissions
    out["sun_tc_u235_burned_g"] = fissions * 235.044 * AMU
    out["sun_tc_u235_burned_mearth"] = fissions * 235.044 * AMU / MEARTH
    out["sun_tc_energy_erg"] = tc["fission_energy_erg"]
    out["fission_energy_per_g_tc"] = tc["fission_energy_erg"] / m_tc
    out["delivery_energy_per_g_1au"] = 0.5 * V_ORB_1AU**2
    out["delivery_over_fission"] = out["delivery_energy_per_g_1au"] / out["fission_energy_per_g_tc"]
    eps = 10 ** (tc["log_eps"] - 12)
    out["sun_tc_number_ratio"] = eps
    out["sun_tc_dmu_over_mu"] = eps * hosts.NUCLIDES["99Tc"]["A"] / MU_ION
    pm = g["Pm II"][KEY]
    out["sun_pm_resupply_g_per_yr"] = pm["resupply_g_per_yr"]
    out["sun_pm_resupply_power_erg_s"] = pm["resupply_power_erg_s"]
    # 147Pm -> 147Sm: steady Pm inventory N decays at N ln2 / t_half per year
    t_half = hosts.NUCLIDES["147Pm"]["t_half"]
    eps_pm = 10 ** (pm["log_eps"] - 12)
    eps_sm_sun = 10 ** (LOG_EPS_SM_SUN - 12)
    out["sun_pm_years_to_double_sm"] = eps_sm_sun / (eps_pm * np.log(2) / t_half)
    out["sun_pm_sm_added_after_1000yr_dex"] = float(np.log10(1 + eps_pm * np.log(2) / t_half * 1000 / eps_sm_sun))
    # White dwarfs fed against settling (Koester 2009, DA, log g = 8, Fe timescales), 0.6 Msun:
    # upkeep power per unit number abundance eps of 99Tc in the mixed layer (pure hydrogen, X = 1).
    msun = 1.98841e33
    e_per_g = hosts.energy_per_atom("99Tc capture") / (hosts.NUCLIDES["99Tc"]["A"] * AMU)
    for name, logq, logtau in (("wd20000", -15.719, -1.68), ("wd8000", -9.002, 2.95)):
        mmix = 0.6 * msun * 10**logq
        flow_h = mmix / (10**logtau * YR)  # g of hydrogen per s through the mixed layer
        out[f"{name}_upkeep_W_per_eps"] = flow_h * hosts.NUCLIDES["99Tc"]["A"] / 1.00794 * e_per_g / 1e7
        am = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]["Am"]["species"]["Tc II"]
        p_am = am["2.4 m space UV, R = 114,000|300"]["upkeep_best_W"]
        out[f"{name}_log_eps_to_match_am"] = float(np.log10(p_am / out[f"{name}_upkeep_W_per_eps"]) + 12)
    for k, v in out.items():
        print(f"{k:36s} {v:.3e}")
    (ROOT / "research" / "derived.json").write_text(json.dumps(out, indent=1))
    _ = YR


if __name__ == "__main__":
    main()
