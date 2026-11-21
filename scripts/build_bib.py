"""Rebuild paper/references.bib from research/bibcodes.txt (key, ADS bibcode) via the SciX API.

Entries that are not in ADS (the IAEA and NIST databases) are appended from EXTRA below.

    uv run python scripts/build_bib.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIX = Path.home() / ".claude" / "skills" / "scix" / "scripts" / "scix.py"

EXTRA = r"""
@MISC{IAEA2026,
       author = {{IAEA Nuclear Data Section}},
        title = "{Live Chart of Nuclides: ground-state half-lives and cumulative fission yields}",
 howpublished = {\url{https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html}},
         year = 2026,
         note = {Accessed 2026 October 28}
}

@MISC{NIST2024,
       author = {{Kramida}, A. and {Ralchenko}, Yu. and {Reader}, J. and {NIST ASD Team}},
        title = "{NIST Atomic Spectra Database (version 5.12)}",
 howpublished = {\url{https://physics.nist.gov/asd}},
          doi = {10.18434/T4W30F},
         year = 2024,
         note = {Accessed 2026 October 28}
}
"""


def main():
    pairs = [ln.split() for ln in (ROOT / "research" / "bibcodes.txt").read_text().splitlines() if ln.strip()]
    key_of = {b: k for k, b in pairs}
    raw = subprocess.run([sys.executable, str(SCIX), "export", *key_of, "-f", "bibtex"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    (ROOT / "research" / "ads_export_raw.bib").write_text(raw, encoding="utf-8")
    out, seen = [], set()
    for e in re.split(r"\n(?=@)", raw.strip()):
        m = re.search(r"adsurl\s*=\s*\{https://ui.adsabs.harvard.edu/abs/([^/}]+)", e)
        b = m.group(1).replace("%26", "&") if m else None
        if b not in key_of:
            continue
        key = key_of[b]
        out.append(re.sub(r"^(@\w+\{)[^,]+,", lambda g, key=key: g.group(1) + key + ",", e, count=1))
        seen.add(b)
    missing = [k for b, k in key_of.items() if b not in seen]
    if missing:
        raise SystemExit(f"not exported: {missing}")
    (ROOT / "paper" / "references.bib").write_text("\n\n".join(out) + "\n" + EXTRA, encoding="utf-8")
    print(f"wrote {len(out)} ADS entries + 2 extra to paper/references.bib")


if __name__ == "__main__":
    main()
