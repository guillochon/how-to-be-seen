#!/usr/bin/env bash
# Download the public inputs that are too large to keep in git (about 45 MB).
#   bash scripts/fetch_inputs.sh
# Kurucz: ATLAS9 grids (Castelli & Kurucz 2003), line lists, Kitt Peak solar flux atlas (Kurucz 2005).
# Mamajek: dwarf colour/Teff sequence (Pecaut & Mamajek 2013, table v2022.04.16).
# IAEA: half-lives and fission yields (Live Chart of Nuclides API).
set -euo pipefail
cd "$(dirname "$0")/.."
K=research/kurucz
mkdir -p "$K/atm" "$K/sun" research/mamajek research/iaea
get() { [ -s "$2" ] || curl -sSfL -m 600 -o "$2" "$1"; echo "ok $2"; }
get http://kurucz.harvard.edu/grids/gridp00odfnew/ap00k2odfnew.dat "$K/atm/ap00k2odfnew.dat"
get http://kurucz.harvard.edu/grids/gridp00odfnew/asunodfnew.dat "$K/atm/asunodfnew.dat"
get http://kurucz.harvard.edu/sun/fluxatlas2005/solarfluxintwl.asc "$K/sun/solarfluxintwl.asc"
get http://kurucz.harvard.edu/linelists/gfall/gf2600.all "$K/gf2600.all"
get http://kurucz.harvard.edu/linelists/gfall/gf2601.all "$K/gf2601.all"
get http://kurucz.harvard.edu/atoms/4300/gf4300.pos "$K/gf4300.pos"
get http://kurucz.harvard.edu/atoms/4301/gf4301.pos "$K/gf4301.pos"
get https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt research/mamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
get "https://nds.iaea.org/relnsd/v1/data?fields=cumulative_fy&parents=235U" research/iaea/fy_u235.csv
get "https://nds.iaea.org/relnsd/v1/data?fields=cumulative_fy&parents=239Pu" research/iaea/fy_pu239.csv
for n in 99tc 97tc 98tc 147pm 145pm; do
  get "https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=$n" "research/iaea/gs_$n.csv"
done
get http://kurucz.harvard.edu/linelists/gfall/gf2401.all "$K/gf2401.all"
get http://kurucz.harvard.edu/linelists/gfall/gf2001.all "$K/gf2001.all"
mkdir -p research/nist
get "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Ra+II&limits_type=0&low_w=&upp_w=&unit=1&submit=Retrieve+Data&de=0&format=1&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on" research/nist/ra2_lines.html
get "https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=226ra" research/iaea/gs_226ra.csv
# Radiative levitation (beacon/levitation.py): surface fluxes, neighbour line lists, partition functions.
get http://kurucz.harvard.edu/grids/gridp00odfnew/fp00k2odfnew.pck "$K/fp00k2odfnew.pck"
mkdir -p "$K/gfall" "$K/partfn"
# 5801-6401: the ionized rare earths of the Ap-star velocity mask (scripts/tc_limits.py).
for s in 0601 1200 1201 1301 1401 1402 2001 2101 2201 2301 2400 2401 2402 2500 2501 2502 2600 2601 2602 2700 2701 2801 2802 5801 5901 6001 6401 ; do
  get "http://kurucz.harvard.edu/linelists/gfall/gf$s.all" "$K/gfall/gf$s.all"
done
for s in 0601 0602 1200 1201 1202 1300 1301 1302 1400 1401 1402 2000 2001 2002 2100 2101 2102 2200 2201 2202 2300 2301 2302 2400 2401 2402 2500 2501 2502 2600 2601 2602 2700 2701 2702 2800 2801 2802 4300 4301 4302 ; do
  get "http://kurucz.harvard.edu/atoms/$s/partfn$s.dat" "$K/partfn/partfn$s.dat"
done
[ -s "$K/tc/tc0_lines.csv" ] || python scripts/fetch_tc_lines.py
