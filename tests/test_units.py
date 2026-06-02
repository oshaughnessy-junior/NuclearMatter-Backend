"""Unit-conversion sanity checks."""
import math
from nmbackend import constants as c


def test_gmsun_km():
    # GMsun/c^2 should be ~1.4766 km
    assert abs(c.GMSUN_KM - 1.4766) < 1e-3


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
