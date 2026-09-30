"""Isothermal-star scan (NSCool_iso.out) and the isothermal cooling curve."""
import numpy as np
import pytest

from nmbackend.cooling import (GapModel, IsothermalStar, NSCool, default_home,
                               isothermal_curve, parse_isothermal)
from nmbackend.cooling.isothermal import YEAR_S

_OUT = """ nmbackend sx_p1s0 sx_n3p2 =   1.0000000000000000        1.0000000000000000
nmbackend isothermal scan
star M R e^phi_s =   1.39999970E+000  1.14302202E+006  7.97792095E-001
eta used by fteff =   2.25733634E-009
columns: Ttil C Lnu Lheat Lgam Teff_inf Lnu_durca Lnu_murca_brem Lnu_pbf Lnu_crust C_core
ISO   1.00000000E+008  7.28E+037  8.35E+033  0.0E+000  7.90E+032  8.57E+005  5.56957395E-268  1.0E+033  2.0E+033  3.0E+032  6.0E+037
 Done !
"""

iso_built = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file(),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")


def test_parse_isothermal():
    s = parse_isothermal(_OUT)
    assert s.T[0] == 1e8 and s.C[0] == 7.28e37 and s.L_durca[0] == pytest.approx(5.57e-268)
    assert (s.mass, s.radius_cm, s.eta_nscool) == (1.3999997, 1143022.02, 2.25733634e-9)


def test_curve_matches_power_law():
    """C = c T, L = l T^n: t(T) = c/(l (n-2)) (T^(2-n) - T0^(2-n))."""
    T = np.logspace(10, 7, 3001)
    c, l, n = 1e30, 1e-40, 8.0
    one = np.ones_like(T)
    s = IsothermalStar(T, c * T, l * T ** n, 0 * T, 0 * T, one, one, one, one, one, one, T,
                       1.4, 1.2e6, 0.8, 0.0)
    cur = isothermal_curve(s)
    exact = c / (l * (n - 2)) * (cur.T ** (2 - n) - T[0] ** (2 - n)) / YEAR_S
    np.testing.assert_allclose(cur.t_yr, exact, rtol=1e-4)


@pytest.fixture(scope="module")
def nscool():
    return NSCool()


@iso_built
def test_eta_reaches_envelope_divided_by_0443(nscool):
    """NSCool_READ.inc.f divides ETA by 0.443 before fteff_acc (PCY97 A10)."""
    s = nscool.isothermal(mass=1.4, eta=1e-9, T_grid=(1e7, 1e9, 3))
    assert s.eta_nscool == pytest.approx(1e-9 / 0.443, rel=1e-6)


@iso_built
def test_sums_match_full_run_when_isothermal(nscool):
    """APR 1.4 Msun at 1e4-1e5 yr is isothermal to 1% (RESULTS-C1): the scan at
    the full run's T_eff must give its L_nu (redshifted, e^(2 phi) weighted).
    The curve is compared with default-step NSCool, itself 0.002-0.003 dex off
    the zero-step limit at 1-10 kyr; the isothermal curve is 0.002 dex off it."""
    full = nscool.run(mass=1.4)
    s = nscool.isothermal(mass=1.4, T_grid=(1e7, 1e9, 401))
    for t in (1e4, 1e5):
        T = np.interp(np.log(full.at(t)), np.log(s.Teff_inf), s.T)
        L_nu = np.exp(np.interp(T, s.T, np.log(s.L_nu)))
        assert L_nu == pytest.approx(full.at(t, "L_nu_inf"), rel=0.1)
    cur = isothermal_curve(nscool.isothermal(mass=1.4))
    for t in (1e3, 3e3, 1e4):
        assert abs(np.log10(cur.at(t) / full.at(t))) < 0.008


@iso_built
def test_crust_blanket_fixes_light_element_star(nscool):
    """APR 1.4, no gaps, eta = 1e-9: T at 1e10 g/cm3 is ~10% below the core at
    1-10 kyr, so T_b = T_core misses NSCool by > 0.01 dex; carrying L_gamma
    through the crust by conduction (rho_b = rhocore) recovers it.  At 3-10 kyr
    default-step NSCool is +0.003 dex off its zero-step limit and the crust
    curve is within 0.0004 dex of that limit (RESULTS-C1)."""
    kw = dict(mass=1.4, gaps=GapModel(n1s0=0, n3p2=0, p1s0=0), eta=1e-9)
    full = nscool.run(**kw)
    s = nscool.isothermal(rho_b=1.6e14, **kw)
    assert np.all(s.T_in > s.T) and np.all(np.diff(s.T_in) > 0)
    strict, crust = isothermal_curve(s), isothermal_curve(s, blanket=True)
    for t in (3e3, 1e4):
        assert abs(np.log10(strict.at(t) / full.at(t))) > 0.012
        assert abs(np.log10(crust.at(t) / full.at(t))) < 0.006
