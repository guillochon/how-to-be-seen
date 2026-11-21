"""Bound-free absorption by metals, transcribed from Kurucz's ATLAS (atlas7v.for, as used for the ATLAS9
grids of Castelli & Kurucz 2003).

COOLOP: C I, Mg I, Al I, Si I and Fe I photoionization from their low levels (SUBROUTINEs C1OP, MG1OP,
AL1OP, SI1OP, FE1OP). LUKEOP: N I, O I, Mg II, Si II and Ca II (FUNCTIONs N1OP, O1OP, MG2OP, SI2OP, CA2OP).
Departure coefficients are 1 (LTE). Not included: the molecular continua of COOLOP (CH, OH, H2 collisions)
and HOTOP (highly ionized light elements, which matter only shortward of about 1000 A in hot stars).

The routines are kept edge by edge, so each entry below is one threshold of the Fortran: the wavenumber
(cm^-1) above which it absorbs, the statistical weight and energy (cm^-1) of the lower level, and its
cross-section (cm^2) as a function of wavenumber.
"""

from __future__ import annotations

import numpy as np

from .const import C, H, K

HC_K = H * C / K
RYD = 109732.298


def _pow(x0, s, p):
    return lambda w: s * (x0 / w) ** p


def _zero(w):
    return 0.0


def _seaton(f0, xsect, power, a):
    """ATLAS FUNCTION SEATON, in frequency (Hz)."""
    def f(freq):
        r = f0 / freq
        return xsect * (a + (1.0 - a) * r) * np.sqrt(r ** int(2.0 * power + 0.01))
    return f


def _c1_p2(w0, terms):
    """C I 2p2 levels: the Luo & Pradhan (1989) background plus Burke & Taylor (1979) resonances, /3."""
    def f(w):
        x = 10.0 ** (-16.80 - (w - w0) / 3.0 / RYD)
        for e0, wd, a, b in terms:
            eps = (w - e0) * 2.0 / wd
            x = x + (a * eps + b) / (eps**2 + 1.0)
        return x / 3.0
    return f


_C1_1S = _c1_p2(69172.400, [(97700.0, 2743.0, 68e-18, 118e-18)])
_C1_1D = _c1_p2(80627.760, [(93917.0, 9230.0, 22e-18, 26e-18), (111130.0, 2743.0, -10.5e-18, 46e-18)])
_C1_3P = _c1_p2(90777.000, [])

# (threshold wavenumber, g, E_low, sigma(waveno))
C1 = [
    (22006.370, 3, 68856.33, _pow(22006.370, 2.1e-18, 1.5)),
    (28880.880, 3, 61981.82, _pow(28880.880, 1.54e-18, 1.2)),
    (30489.700, 9, 60373.00, _pow(30489.700, 0.2e-18, 1.2)),
    (69172.400, 1, 21648.02, _C1_1S),
    (69235.820, 2, 21648.02, _C1_1S),  # the 2P1.5 limit doubles the 2P0.5 term
    (69767.350, 15, 64088.85, _pow(69767.350, 16e-18, 3)),
    (80627.760, 5, 10192.66, _C1_1D),
    (80691.180, 10, 10192.66, _C1_1D),
    (90777.000, 5, 43.42, _C1_3P),
    (90804.000, 3, 16.42, _C1_3P),
    (90820.420, 1, 0.0, _C1_3P),
    (90840.420, 10, 43.42, _C1_3P),
    (90867.420, 6, 16.42, _C1_3P),
    (90883.840, 2, 0.0, _C1_3P),
    (100121.000, 5, 33735.20, _pow(100121.000, 1e-18, 3)),
]


def _mg1_3p1p(w):
    return 16e-18 * (26619.756 / w) ** 2.1 - 7.8e-18 * (26619.756 / w) ** 9.5


def _mg1_3p3p(w):
    return max(20e-18 * (39759.842 / w) ** 2.7, 40e-18 * (39759.842 / w) ** 14)


MG1 = [
    (13713.986, 15, 47957.034, _pow(13713.986, 25e-18, 2.7)),
    (13823.223, 9, 47847.797, _pow(13823.223, 33.8e-18, 2.8)),
    (15267.955, 5, 46403.065, _pow(15267.955, 45e-18, 2.7)),
    (18167.687, 1, 43503.333, _pow(18167.687, 0.43e-18, 2.6)),
    (20473.617, 3, 41197.043, _pow(20473.617, 2.1e-18, 2.6)),
    (26619.756, 3, 35051.264, _mg1_3p1p),
    (39759.842, 5, 21911.178, _mg1_3p3p),
    (39800.556, 3, 21870.464, _mg1_3p3p),
    (39820.615, 1, 21850.405, _mg1_3p3p),
    (61671.020, 1, 0.0, _pow(61671.020, 1.1e-18, 10)),
]

AL1 = [
    (8002.467, 6, 40275.903, _pow(8002.467, 50e-18, 3)),
    (9346.231, 10, 38932.139, _pow(9346.231, 50e-18, 3)),
    (10588.957, 2, 37689.413, _pow(10588.957, 56.7e-18, 1.9)),
    (15318.007, 6, 32960.363, _pow(15318.007, 14.5e-18, 1)),
    (15842.129, 10, 32436.241, _pow(15842.129, 47e-18, 1.83)),
    (22930.614, 2, 25347.756, _pow(22930.614, 10e-18, 2)),
    (48166.309, 4, 112.061, _pow(48166.309, 65e-18, 5)),
    (48278.370, 2, 0.0, _pow(48166.309, 65e-18, 5)),
    (55903.260, 12, 29097.11, _pow(55903.260, 10e-18, 2)),
]


def _si1_reson(w0, wd, a, b):
    def f(w):
        eps = (w - w0) * 2.0 / wd
        return (a * eps + b) / (eps**2 + 1.0)
    return f


_SI1_R1S = _si1_reson(70000.0, 6500.0, 97e-18, 94e-18)
_SI1_R1D = _si1_reson(78600.0, 13000.0, -10e-18, 77e-18)


def _si1_3p(w):
    x = 72e-18 * (65524.393 / w) ** 1.90
    if w > 74000.0:
        x = 93e-18 * (65524.393 / w) ** 4.00
    return x / 3.0


SI1 = [
    (17777.641, 15, 48161.459, _pow(17777.641, 18e-18, 3)),
    (24947.216, 3, 40991.884, _pow(24947.216, 4.09e-18, 2)),
    (26079.180, 9, 39859.920, _pow(26079.180, 1.25e-18, 2)),
    (50353.180, 1 / 3, 15394.370, lambda w: 37e-18 * (50353.180 / w) ** 2.40 + _SI1_R1S(w)),
    (50640.630, 2 / 3, 15394.370, lambda w: 37e-18 * (50640.630 / w) ** 2.40 + _SI1_R1S(w)),
    (59448.700, 5 / 3, 6298.850, lambda w: 24.5e-18 * (59448.700 / w) ** 1.85 + _SI1_R1D(w)),
    (59736.150, 10 / 3, 6298.850, lambda w: 24.5e-18 * (59736.150 / w) ** 1.85 + _SI1_R1D(w)),
    (63446.510, 15, 45303.310, _pow(63446.510, 18e-18, 3)),
    (65524.393, 5, 223.157, _si1_3p),
    (65670.435, 3, 77.115, _si1_3p),
    (65747.550, 1, 0.0, _si1_3p),
    (65811.843, 10, 223.157, _si1_3p),
    (65957.885, 6, 77.115, _si1_3p),
    (66035.000, 2, 0.0, _si1_3p),
    (75423.767, 5, 33326.053, _pow(75423.767, 15e-18, 3)),
]

# FE1OP: 48 pseudo-levels with one smoothed edge each.
_FE_G = np.array([25., 35., 21., 15., 9., 35., 33., 21., 27., 49., 9., 21., 27., 9., 9., 25., 33., 15., 35., 3., 5., 11.,
                  15., 13., 15., 9., 21., 15., 21., 25., 35., 9., 5., 45., 27., 21., 15., 21., 15., 25., 21., 35., 5.,
                  15., 45., 35., 55., 25.])
_FE_E = np.array([500., 7500., 12500., 17500., 19000., 19500., 19500., 21000., 22000., 23000., 23000., 24000., 24000.,
                  24500., 24500., 26000., 26500., 26500., 27000., 27500., 28500., 29000., 29500., 29500., 29500., 30000.,
                  31500., 31500., 33500., 33500., 34000., 34500., 34500., 35000., 35500., 37000., 37000., 37000., 38500.,
                  40000., 40000., 41000., 41000., 43000., 43000., 43000., 43000., 44000.])
_FE_W = np.array([63500., 58500., 53500., 59500., 45000., 44500., 44500., 43000., 58000., 41000., 54000., 40000., 40000.,
                  57500., 55500., 38000., 57500., 57500., 37000., 54500., 53500., 55000., 34500., 34500., 34500., 34000.,
                  32500., 32500., 32500., 32500., 32000., 29500., 29500., 31000., 30500., 29000., 27000., 54000., 27500.,
                  24000., 47000., 23000., 44000., 42000., 42000., 21000., 42000., 42000.])

# Si II from Peach (ATLAS SI2OP): ln(sigma * U) on a grid of T and ln(frequency).
_SI2_PEACH = np.array([
    [-43.8941, -43.8941, -43.8941, -43.8941, -43.8941, -43.8941],
    [-42.2444, -42.2444, -42.2444, -42.2444, -42.2444, -42.2444],
    [-40.6054, -40.6054, -40.6054, -40.6054, -40.6054, -40.6054],
    [-54.2389, -52.2906, -50.8799, -49.8033, -48.9485, -48.2490],
    [-50.4108, -48.4892, -47.1090, -46.0672, -45.2510, -44.5933],
    [-52.0936, -50.0741, -48.5999, -47.4676, -46.5649, -45.8246],
    [-51.9548, -49.9371, -48.4647, -47.3340, -46.4333, -45.6947],
    [-54.2407, -51.7319, -49.9178, -48.5395, -47.4529, -46.5709],
    [-52.7355, -50.2218, -48.4059, -47.0267, -45.9402, -45.0592],
    [-53.5387, -50.9189, -49.0200, -47.5750, -46.4341, -45.5082],
    [-53.2417, -50.6234, -48.7252, -47.2810, -46.1410, -45.2153],
    [-53.5097, -50.8535, -48.9263, -47.4586, -46.2994, -45.3581],
    [-54.0561, -51.2365, -49.1980, -47.6497, -46.4302, -45.4414],
    [-53.8469, -51.0256, -48.9860, -47.4368, -46.2162, -45.2266],
])
_SI2_FREQ = np.array([4.9965417e15, 3.9466738e15, 1.5736321e15, 1.5171539e15, 9.2378947e14, 8.3825004e14, 7.6869872e14])
_SI2_FLOG = np.array([36.32984, 36.14752, 35.91165, 34.99216, 34.95561, 34.45951, 34.36234, 34.27572, 34.20161])
_SI2_TLG = np.array([9.21034, 9.39266, 9.54681, 9.68034, 9.79813, 9.90349])


def _si2op(freq: float, T: np.ndarray) -> np.ndarray:
    """Si II cross-section times partition function (cm^2), ATLAS SI2OP."""
    n = next((i for i, f in enumerate(_SI2_FREQ) if freq > f), 7) + 1  # Fortran index 1..8
    d = (np.log(freq) - _SI2_FLOG[n - 1]) / (_SI2_FLOG[n] - _SI2_FLOG[n - 1])
    if n > 2:
        n = 2 * n - 2
    if n == 14:
        n = 13
    x = _SI2_PEACH[n] * d + _SI2_PEACH[n - 1] * (1.0 - d)  # 6 temperatures
    nt = np.clip((T / 2000.0).astype(int) - 4, 1, 5)  # Fortran 1..5
    dt = (np.log(T) - _SI2_TLG[nt - 1]) / (_SI2_TLG[nt] - _SI2_TLG[nt - 1])
    return np.exp(x[nt - 1] * (1.0 - dt) + x[nt] * dt) * 6.0


def _sum_edges(edges, waveno: float, T: np.ndarray) -> np.ndarray:
    tot = np.zeros_like(T)
    for w0, g, e, sig in edges:
        if waveno < w0:
            continue
        s = sig(waveno)
        if s:
            tot = tot + s * g * np.exp(-e * HC_K / T)
    return tot


def kappa_metals(st, wl_A: float) -> np.ndarray:
    """Metal bound-free absorption coefficient per unit volume (cm^-1), stimulated emission included."""
    from .levitation import U, ion_fractions

    m = st.m
    T, ne = m.T, m.ne
    freq = C / (wl_A * 1e-8)
    waveno = 1e8 / wl_A
    stim = 1.0 - np.exp(-H * freq / (K * T))
    tkev = K * T / 1.602176634e-12

    def nfp(z, ion):  # number density over partition function
        n = st.nH * 10.0 ** m.log_abund[z] * ion_fractions(z, T, ne)[ion]
        return n / U(z, ion, T)

    k = np.zeros_like(T)
    for z, edges in ((6, C1), (12, MG1), (13, AL1), (14, SI1)):
        if waveno >= edges[0][0]:
            k = k + _sum_edges(edges, waveno, T) * nfp(z, 0)
    if waveno >= 21000.0:
        sel = _FE_W <= waveno
        xs = 3e-18 / (1.0 + ((_FE_W[sel] + 3000.0 - waveno) / _FE_W[sel] / 0.1) ** 4)
        k = k + np.sum((xs * _FE_G[sel])[:, None] * np.exp(-_FE_E[sel][:, None] * HC_K / T[None, :]), axis=0) * nfp(26, 0)
    # LUKEOP
    n1 = 0.0
    if freq >= 3.517915e15:
        n1 = n1 + _seaton(3.517915e15, 1.142e-17, 2.0, 4.29)(freq) * 4.0
    if freq >= 2.941534e15:
        n1 = n1 + _seaton(2.941534e15, 4.41e-18, 1.5, 3.85)(freq) * 10.0 * np.exp(-2.384 / tkev)
    if freq >= 2.653317e15:
        n1 = n1 + _seaton(2.653317e15, 4.2e-18, 1.5, 4.34)(freq) * 6.0 * np.exp(-3.575 / tkev)
    o1 = _seaton(3.28805e15, 2.94e-18, 1.0, 2.66)(freq) * 9.0 if freq >= 3.28805e15 else 0.0
    mg2 = 0.0
    if freq >= 3.635492e15:
        mg2 = mg2 + _seaton(3.635492e15, 1.40e-19, 4.0, 6.7)(freq) * 2.0
    if freq >= 2.564306e15:
        mg2 = mg2 + 5.11e-19 * (2.564306e15 / freq) ** 3 * 6.0 * np.exp(-4.43 / tkev)
    ca2 = 0.0
    if freq >= 2.870454e15:
        ca2 = ca2 + 5.4e-20 * (2.870454e15 / freq) ** 3 * 2.0
    if freq >= 2.460127e15:
        ca2 = ca2 + 1.64e-17 * np.sqrt(2.460127e15 / freq) * 10.0 * np.exp(-1.697 / tkev)
    if freq >= 2.110779e15:
        ca2 = ca2 + _seaton(2.110779e15, 4.13e-18, 3.0, 0.69)(freq) * 6.0 * np.exp(-3.142 / tkev)
    k = k + (n1 * nfp(7, 0) + o1 * nfp(8, 0) + mg2 * nfp(12, 1) + _si2op(freq, T) * nfp(14, 1) + ca2 * nfp(20, 1))
    return k * stim
