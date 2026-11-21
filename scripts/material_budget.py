"""What the bred beacons of Table 3 consume besides energy: fissile fuel and target, the cost of other neutron
sources, and the energy of delivering the product.

Fuel. Breeding N atoms by one capture each takes N / (nu_bar - 1) thermal fissions of 235U at best
(beacon/hosts.py). We give the 235U burned, the natural uranium that holds it if only 235U fissions, and the
uranium consumed if a breeder fissions 238U as well (one 238 u nucleus per fission).
Target. N atoms of 98Mo (96Ru, 144Sm), and the natural element that holds them (IUPAC isotopic abundances).
Both for building the inventory once and for one half-life of steady upkeep (ln 2 of the inventory).

Neutron sources. Energy released per spare neutron: thermal fission 200 MeV / 1.46 = 137 MeV; catalyzed D-D fusion,
in which the T and 3He of the two D-D branches burn with D (6 D -> 2 4He + 2 p + 2 n + 43.2 MeV), 21.6 MeV, with
no neutron needed to breed fuel; D-T fusion releases 17.6 MeV per neutron, but that neutron must breed the next
triton, so its spares come only from neutron multiplication.

Delivery. Energy per kilogram of product, compared with the 137 MeV per atom of breeding it: (a) cancelling an orbital
speed v_orb with a rocket of exhaust speed v_e (energy of the exhaust, ideal efficiency); (b) carrying it to another
star at 0.01 c and stopping it there with a rocket of v_e = 0.03 c.

    uv run python scripts/material_budget.py     # writes research/material_budget.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, hosts  # noqa: E402
from beacon.const import AMU, EV  # noqa: E402

ROWS = [("Sun", "G", "Tc I", "8.2 m, R = 140,000"), ("Am star", "Am", "Tc II", "2.4 m space UV, R = 114,000"),
        ("Am star", "Am", "Tc II", "8.2 m UV-blue, R = 80,000"), ("Late B", "B", "Tc II", "2.4 m space UV, R = 114,000"),
        ("Late B", "B", "Tc II", "8.2 m UV-blue, R = 80,000"), ("K dwarf", "K", "Tc I", "8.2 m, R = 140,000")]
U235_FRAC = 0.007204  # natural abundance of 235U (IUPAC)
TARGET = {"98Mo": (97.905, 0.2439), "96Ru": (95.908, 0.0554), "144Sm": (143.912, 0.0307)}  # mass (u), abundance
E_SPARE = {"235U fission": 200.0 / (hosts.NU_BAR - 1.0), "catalyzed D-D fusion": 43.2 / 2.0}  # MeV per spare neutron
C_KMS = 2.998e5


def rocket_energy_per_kg(dv_kms: float, ve_kms: float) -> float:
    """Exhaust kinetic energy (J) per kg of payload for a velocity change dv (ideal rocket)."""
    prop = np.expm1(dv_kms / ve_kms)
    return 0.5 * prop * (ve_kms * 1e3) ** 2


def main():
    res = json.loads((ROOT / "research" / "results.json").read_text())["hosts"]
    breed_j_per_kg = E_SPARE["235U fission"] * 1e6 * EV * 1e-7 / (hosts.NUCLIDES["99Tc"]["A"] * AMU * 1e-3)
    out = dict(breed_J_per_kg=breed_j_per_kg, e_spare_MeV=E_SPARE,
               dK_catalyzed_DD=-np.log10(E_SPARE["235U fission"] / E_SPARE["catalyzed D-D fusion"]) / 10, rows=[])
    for label, hk, sp, inst in ROWS:
        r = res[hk]["species"][sp][f"{inst}|300"]
        m_kg = r["mass_g"] / 1e3
        n = m_kg / (hosts.NUCLIDES["99Tc"]["A"] * AMU * 1e-3)
        fis = n / (hosts.NU_BAR - 1.0)
        u235 = fis * 235.044 * AMU * 1e-3
        a, f = TARGET["98Mo"]
        rec = dict(host=label, species=sp, instrument=inst, inventory_kg=m_kg, atoms=n, fissions=fis, u235_kg=u235,
                   natural_u_kg=u235 / U235_FRAC, breeder_u_kg=fis * 238.05 * AMU * 1e-3,
                   mo98_kg=n * a * AMU * 1e-3, natural_mo_kg=n * a * AMU * 1e-3 / f, per_half_life_factor=np.log(2))
        out["rows"].append(rec)
        print(f"{label:8s} {sp:6s} {inst[:14]:14s} inventory {m_kg:.1e} kg: 235U {u235:.1e} kg, natural U {rec['natural_u_kg']:.1e} kg "
              f"(breeder {rec['breeder_u_kg']:.1e}), natural Mo {rec['natural_mo_kg']:.1e} kg")
    out["delivery"] = {
        "orbit 30 km/s, v_e 4.5 km/s": rocket_energy_per_kg(30.0, 4.5),
        "orbit 30 km/s, v_e 30 km/s": rocket_energy_per_kg(30.0, 30.0),
        "orbit 30 km/s, kinetic energy only": 0.5 * 30e3**2,
        "interstellar 0.01 c and stop, v_e 0.03 c": rocket_energy_per_kg(2 * 0.01 * C_KMS, 0.03 * C_KMS),
        "interstellar 0.01 c, kinetic energy only": 0.5 * (0.01 * C_KMS * 1e3) ** 2,
    }
    print(f"breeding: {breed_j_per_kg:.2e} J/kg; catalyzed D-D lowers K by {-out['dK_catalyzed_DD']:.3f}")
    for k, v in out["delivery"].items():
        print(f"  delivery {k}: {v:.1e} J/kg = {v / breed_j_per_kg:.1e} of breeding")
    (ROOT / "research" / "material_budget.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
