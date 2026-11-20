"""Unit-conversion sanity checks."""
import math
from nmbackend import constants as c


def test_gmsun_km():
    # GMsun/c^2 should be ~1.4766 km
    assert abs(c.GMSUN_KM - 1.4766) < 1e-3


def test_gmsun_km_is_iau_nominal():
    # IAU 2015 B3 nominal (GM)_sun = 1.3271244e20 m^3 s^-2, as in lal 7.7.0
    # include/lal/LALConstants.h: LAL_MRSUN_SI = 1.476625038050124729...e3 m,
    # LAL_MSUN_SI = 1.988409870698050731...e30 kg.  1e-12 separates it from the
    # older LAL values (TDB GM 1.32712442099e20: 1476.62506 m, 1.988409902e30 kg,
    # off by 1.5e-8) and from the pre-2026-10-14 nmbackend value (1477.0039 m).
    assert abs(c.GMSUN_KM * 1e3 / 1.476625038050124729e3 - 1.0) < 1e-12
    assert abs(c.MSUN_SI / 1.988409870698050731e30 - 1.0) < 1e-12
    assert abs(c.G_SI * c.MSUN_SI / c.GM_SUN_SI - 1.0) < 1e-15


def test_mevfm3_to_geom():
    # nuclear saturation energy density ~150 MeV/fm^3 -> ~2e-4 km^-2
    eps_geom = 150.0 * c.MEVFM3_TO_GEOM
    assert 1e-4 < eps_geom < 5e-4


def test_geom_pa_roundtrip():
    # km^-2 -> Pa -> back via MEVFM3 factors should be self-consistent
    p_mevfm3 = 10.0
    p_pa = p_mevfm3 * c.MEVFM3_TO_PA
    p_geom = p_mevfm3 * c.MEVFM3_TO_GEOM
    assert abs(p_geom * c.GEOM_KM2_TO_PA - p_pa) / p_pa < 1e-6
