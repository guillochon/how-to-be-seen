"""Fetch spectra and set Tc limits for many stars of scripts/tc_limits.py STARS, then tabulate the results.

Each star runs in its own process, so a failure is logged and the batch goes on. Stars that already have a
result are skipped unless --redo is given. The summary of all results is research/tc_limits/summary.csv. Its
K_atmosphere assumes that the Tc stays in the atmosphere, and K_host that it fills the mixed mass of the star's
host class in the paper (scripts/tc_limits.py, host_class): the atmosphere of a quiet late-B star, the turbulent
layer of an A star, or the convection zone of an FGK star.

    uv run python scripts/tc_batch.py              # every star in STARS
    uv run python scripts/tc_batch.py hd27295 ...  # these stars
    uv run python scripts/tc_batch.py --shard 2/4  # every fourth star, from the third: run 4 in parallel
    uv run python scripts/tc_batch.py --summary    # only rewrite the summary
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tc_limits as T  # noqa: E402

LOGS = T.OUT / "logs"


def run(script: str, key: str, timeout: float) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{key}.{script}.log").open("w") as log:
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).parent / f"{script}.py"), key], stdout=log,
                               stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\ntimed out after {timeout:.0f} s\n")
            return False
    return p.returncode == 0


def summary() -> Path:
    rows = []
    for key, star in T.STARS.items():
        path = T.OUT / f"{key}.json"
        if not path.exists() or key in T.EXCLUDED:
            continue
        b = json.loads(path.read_text())
        c = b["best"].get("calibrated")
        atm = next((m for m in b["mixed_msun"] if m.startswith("atmosphere")), None)
        k_atm = T.inventory(c["log_eps"], b["mixed_msun"][atm])["sagan_K"] if c and atm else float("nan")
        mhost = T.host_mixed_mass(key, star, b["mixed_msun"])
        k_host = T.inventory(c["log_eps"], mhost)["sagan_K"] if c else float("nan")
        rows.append(dict(key=key, name=star.name, teff=star.teff, logg=star.logg, fe_lo=(star.fe or (None,))[0],
                         instruments=" ".join(f"{k}:{v['n']}" for k, v in b["spectra"].items()),
                         calibration=" ".join(f"{k}:x{v:g}" for k, v in b.get("calibration", {}).items()),
                         log_eps=c["log_eps"] if c else None, blanket=c["blanket"] if c else None,
                         line=c["line"] if c else None, instrument=c["instrument"] if c else None,
                         K_atmosphere=round(k_atm, 3) if k_atm == k_atm else None, host=T.host_class(key, star),
                         K_host=round(k_host, 3) if k_host == k_host else None, source=star.source))
    rows.sort(key=lambda r: (r["log_eps"] is None, r["log_eps"]))
    path = T.OUT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stars", nargs="*")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run only the stars whose index is i modulo n")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    if not a.summary:
        for key in (a.stars or list(T.STARS))[i::n]:
            if (T.OUT / f"{key}.json").exists() and not a.redo:
                continue
            t0 = time.time()
            ok = run("fetch_spectra", key, 7200)  # existing files are skipped
            ok = ok and run("tc_limits", key, 3600) and run("tc_limits_plot", key, 1800)
            print(f"{key:16s} {'ok' if ok else 'FAILED (see logs)'} {time.time() - t0:.0f} s", flush=True)
    print(f"summary: {summary()}")


if __name__ == "__main__":
    main()
