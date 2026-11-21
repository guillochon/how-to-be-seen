"""Physical constants in cgs units (CODATA 2018 values, rounded)."""

C = 2.99792458e10  # speed of light (cm s^-1)
H = 6.62607015e-27  # Planck constant (erg s)
K = 1.380649e-16  # Boltzmann constant (erg K^-1)
ME = 9.1093837e-28  # electron mass (g)
E_CHARGE = 4.80320471e-10  # electron charge (esu)
AMU = 1.66053907e-24  # atomic mass unit (g)
EV = 1.602176634e-12  # electron volt (erg)
KEV = K / EV  # Boltzmann constant (eV K^-1)
SIGMA_T = 6.6524587e-25  # Thomson cross-section (cm^2)
MSUN = 1.98841e33  # solar mass (g)
MEARTH = 5.9722e27  # Earth mass (g)
YR = 3.15576e7  # Julian year (s)
PC = 3.0856776e18  # parsec (cm)
RSUN = 6.957e10  # nominal solar radius (cm)

# pi e^2 / (m_e c): integrated classical line cross-section per unit oscillator strength (cm^2 Hz)
LINE_CROSS = 3.1415926535 * E_CHARGE**2 / (ME * C)
# (2 pi m_e k / h^2)^(3/2), the Saha prefactor per unit T^(3/2) (cm^-3 K^-3/2)
SAHA = (2 * 3.1415926535 * ME * K / H**2) ** 1.5
AU = 1.495978707e13  # astronomical unit (cm)
