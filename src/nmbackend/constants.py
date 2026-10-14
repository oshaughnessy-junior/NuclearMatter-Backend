"""Physical constants and unit conversions for NuclearMatter-Backend.

Internal TOV units are *geometric* (G = c = 1) with **length in km**.  In these
units energy density and pressure have dimension km^-2, mass has dimension km,
and the pseudo-enthalpy h = integral dp/(eps+p) is dimensionless.  Masses are
reported in solar masses by dividing the geometric mass (km) by GMsun/c^2.

Nuclear inputs are in MeV and fm^-3; the conversion ``MEVFM3_TO_GEOM`` maps an
energy density in MeV/fm^3 to geometric km^-2.

Motivated by unit handling in RIFT ``EOSManager`` and ``jester`` (MIT); see
ATTRIBUTION.md.
"""
from __future__ import annotations

# --- fundamental constants (CODATA-ish, SI) ---
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_SI = 2.99792458e8         # m / s
# Solar mass parameter: the IAU 2015 Resolution B3 nominal (GM)_sun (exact by
# definition), as current LALSuite (LAL_GMSUN_SI in lal 7.7.0 LALConstants.h)
# and the Q2C oracle use.  Until 2026-10-14 this module had MSUN_SI = 1.98892e30
# kg (old LAL's solar mass, meant for CODATA-1986 G); paired with CODATA-2018 G,
# GMSUN_KM came out 1.4770039 km, 2.566e-4 above the 1.476625 km its comment
# claimed (docs/devlog/2026-10-14-gmsun-iau.md).  MSUN_SI is derived, so
# G_SI * MSUN_SI == GM_SUN_SI.
GM_SUN_SI = 1.3271244e20    # m^3 s^-2
MSUN_SI = GM_SUN_SI / G_SI  # kg, 1.98840987069805e30 (= lal 7.7 LAL_MSUN_SI)

# --- nuclear constants ---
HBARC = 197.3269804         # MeV fm
M_N = 939.565421            # neutron mass, MeV
M_P = 938.272088            # proton mass, MeV
M_E = 0.510998950           # electron mass, MeV
M_MU = 105.6583755          # muon mass, MeV
M_BARYON = 939.0            # representative baryon (rest-mass density) mass, MeV
MEV_TO_FM_INV = 1.0 / HBARC  # 1 MeV = (1/hbarc) fm^-1

# --- geometric length scale: GMsun/c^2 in km ---
# 1.476625 km (IAU nominal).  Geometric mass [km] / GMSUN_KM -> mass [Msun].
GMSUN_KM = GM_SUN_SI / C_SI**2 / 1.0e3   # 1.4766250 km

# --- energy-density / pressure conversion: MeV/fm^3 -> geometric km^-2 ---
# eps_geom[m^-2] = (G/c^4) * eps_SI[Pa];  1 MeV/fm^3 = 1.602176634e32 Pa.
_MEVFM3_TO_PA = 1.602176634e32
_G_OVER_C4 = G_SI / C_SI**4                    # m^-1 kg^-1 s^2
MEVFM3_TO_GEOM = _G_OVER_C4 * _MEVFM3_TO_PA * 1.0e6   # km^-2 per (MeV/fm^3)

# --- SI helpers for the RIFT adapter (pressure in Pa, density in kg/m^3) ---
GEOM_KM2_TO_PA = 1.0 / (_G_OVER_C4 * 1.0e6)    # km^-2 -> Pa
MEVFM3_TO_PA = _MEVFM3_TO_PA                    # MeV/fm^3 -> Pa
FM3_TO_M3 = 1.0e-45                             # fm^3 -> m^3
NUCLEON_KG = M_BARYON * 1.602176634e-13 / C_SI**2  # kg per baryon (rest mass)

# default nuclear saturation density (fm^-3)
N_SAT_DEFAULT = 0.16
