"""NSCool driver (nmbackend.cooling).  Binary tests skip unless NSCool is built."""
import numpy as np
import pytest

from nmbackend.cooling import (GapModel, NSCool, NSCoolError, default_home,
                               parse_teff_file)

_TEFF = """ EOS
              Mstar     = 1.400E+00    Rstar     = 1.143E+06
    Step      Time            Teff       L_phot       L_nu        L_heat
       1   1.000E-12        6.774E+06   3.081E+36   1.013E+47   0.000E+00
       2   1.000E+03        1.000E+06   2.000E+33   3.000E+30   0.000E+00
"""

built = pytest.mark.skipif(not (default_home() / "Code" / "NSCool.out").is_file(),
                           reason="NSCool not built (nmbackend/cooling/build_nscool.sh)")


def test_pairing_file_carries_scalings():
    lines = GapModel(p1s0=3, sx_p1s0=1.5, sy_p1s0=0.7).pairing_file().splitlines()
    assert [l.split()[1] for l in lines[1:]] == [
        "SFN1S0", "SFN3P2", "SFP1S0", "SFL1S0", "FN1S0", "FN3P2", "FP1S0",
        "FL1S0", "SFQUARK", "SXP1S0", "SXN3P2"]
    vals = {l.split()[1]: float(l.split()[0]) for l in lines[1:]}
    assert (vals["FP1S0"], vals["SXP1S0"], vals["SFP1S0"]) == (0.7, 1.5, 3.0)
    with pytest.raises(ValueError):
        GapModel(sx_n3p2=0.0)


def test_parse_teff_file(tmp_path):
    f = tmp_path / "Teff.dat"
    f.write_text(_TEFF)
    c = parse_teff_file(f)
    assert c.mass == 1.4 and c.radius_cm == 1.143e6
    np.testing.assert_array_equal(c.t_yr, [1e-12, 1e3])
    # e^(2 phi_s) = 1 - 2GM/(Rc^2)
    assert c.exp_phi_s == pytest.approx(np.sqrt(1 - 2 * 1.47663e5 * 1.4 / 1.143e6), rel=1e-5)
    assert c.Ts[1] == pytest.approx(1e6 / c.exp_phi_s)
    assert c.at(1e3) == pytest.approx(1e6)


@pytest.fixture(scope="module")
def nscool():
    return NSCool()


@pytest.fixture(scope="module")
def reference(nscool):
    """NSCool's bundled output for Model_1 (APR, 1.4 Msun, SFB / gap a / T73)."""
    return parse_teff_file(nscool.home / "Model_1" / "Teff_Try_KEEP.dat")


@built
def test_reproduces_bundled_model1(nscool, reference):
    c = nscool.run(profile=nscool.home / "TOV/Profile/Prof_APR_Cat_1.4.dat")
    assert len(c.t_yr) == len(reference.t_yr)
    for k in ("t_yr", "Teff_inf", "L_gamma_inf", "L_nu_inf"):
        np.testing.assert_allclose(getattr(c, k), getattr(reference, k), rtol=1e-3)


@built
def test_tov_route_matches_bundled_profile(nscool, reference):
    c = nscool.run(mass=1.4)
    ages = np.logspace(0, 6, 13)
    np.testing.assert_allclose(c.at(ages), reference.at(ages), rtol=1e-3)


@built
def test_gap_scaling_follows_zhou_eq10(nscool):
    """Tc(n) = s_y Tc_0(n / s_x) for p1S0 and n3P2, zone by zone."""
    b = nscool.run(mass=1.4, star=True).star
    s = nscool.run(mass=1.4, star=True,
                   gaps=GapModel(sx_p1s0=2.0, sy_p1s0=1.5, sx_n3p2=0.8)).star
    nb, o = b["nb"], np.argsort(b["nb"])
    tcp = 1.5 * np.interp(nb / 2.0, nb[o], b["Tcp"][o])
    m = (tcp > 1e8) & (nb / 2.0 > nb.min())
    assert m.sum() > 20
    np.testing.assert_allclose(s["Tcp"][m], tcp[m], rtol=0.02)
    tcn = np.interp(nb / 0.8, nb[o], b["Tcn"][o], right=np.nan)
    m = np.isfinite(tcn) & (tcn > 1e7)
    assert m.sum() > 20
    np.testing.assert_allclose(s["Tcn"][m], tcn[m], rtol=0.02)


@built
def test_light_elements_raise_early_teff(nscool, reference):
    c = nscool.run(mass=1.4, eta=1e-7)
    assert c.at(1e3) > 1.05 * reference.at(1e3)


@built
def test_unreachable_mass_raises(nscool):
    with pytest.raises(NSCoolError):
        nscool.make_profile(nscool.default_eos, 3.5, "unused.dat")
