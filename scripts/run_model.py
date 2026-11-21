"""Compute equivalent-width curves for all hosts and species and cache them.

Writes research/curves.pkl (read by scripts/results.py and the figure scripts).

    uv run python scripts/run_model.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beacon import ROOT, hosts, survey  # noqa: E402


def main():
    out = {}
    for h in hosts.host_grid():
        print("host", h.key, flush=True)
        out[h.key] = survey.build_curves(h)
    with (ROOT / "research" / "curves.pkl").open("wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
