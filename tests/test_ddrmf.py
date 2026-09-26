"""Density-dependent RMF (Typel-Wolter): DD2 and DDME2 reference checks."""
import math

import numpy as np
import jax.numpy as jnp
import pytest

from nmbackend.eos import DDRMFEOS, DD2, DDME2
from nmbackend.eos.ddrmf import (_eps_p, _solve_xp, couplings_from_saturation_dd,
                                 saturation_properties, symmetric_matter)
from nmbackend.durca import durca_onset, durca_mass
from nmbackend.tov import build_family, default_hc_grid


def _star(params, n_models=80):
    m = DDRMFEOS()
    tab, comp = m.eos_table(params), m.composition(params)
    fam = build_family(tab, default_hc_grid(tab, n_models=n_models))
    M = np.asarray(fam.M_msun)
    i = int(np.argmax(np.where(np.asarray(fam.stable), M, -np.inf)))
    on = durca_onset(comp)
    return M[i], on, durca_mass(tab, on.n_DU, fam.h_c[i]), comp


# Fortin et al., PRC 94, 035804 (2016), arXiv:1604.01944, Table II.
@pytest.mark.parametrize("params, ref", [
    (DD2, dict(n0=0.149, BA=-16.0, K=242.6, E_sym=31.7, L=55.0)),
    (DDME2, dict(n0=0.152, BA=-16.1, K=250.9, E_sym=32.3, L=51.2)),
])
def test_saturation_properties(params, ref):
    s = saturation_properties(params)
    tol = dict(n0=0.001, BA=0.06, K=0.5, E_sym=0.06, L=0.2)
    for k, v in ref.items():
        assert abs(s[k] - v) < tol[k], (k, s[k])


def test_dd2_effective_mass():
    """Typel et al. 2010 (arXiv:0908.2344): Dirac m* = 0.5625 m_nuc."""
    assert abs(saturation_properties(DD2)["mstar_frac"] - 0.5625) < 0.001


@pytest.mark.parametrize("params, mmax", [(DD2, 2.42), (DDME2, 2.48)])
def test_maximum_mass(params, mmax):
    """Fortin et al. 2016, Table II (nucleonic M_max)."""
    assert abs(_star(params)[0] - mmax) < 0.03


def test_rearrangement_gives_thermodynamic_pressure():
    """p = n^2 d(eps/n)/dn in symmetric matter and in beta equilibrium; this
    fails by tens of MeV/fm^3 without the rearrangement term."""
    n = np.linspace(0.1, 1.0, 901)
    s = symmetric_matter(n, DD2)
    ea = np.asarray(s["EA"])
    np.testing.assert_allclose(np.asarray(s["p"])[1:-1],
                               (n**2 * np.gradient(ea, n))[1:-1], atol=2e-3)
    nj = jnp.asarray(n)
    xp = _solve_xp(nj, DD2)
    eps, pres = (np.asarray(a) for a in _eps_p(nj, xp, DD2))
    np.testing.assert_allclose(pres[1:-1], (n**2 * np.gradient(eps / n, n))[1:-1],
                               rtol=2e-3, atol=2e-2)


def test_inversion_recovers_dd2():
    s = saturation_properties(DD2)
    c = couplings_from_saturation_dd(
        n0=s["n0"], BA=s["BA"], K=s["K"], mstar_frac=s["mstar_frac"],
        J=s["E_sym"], L=s["L"], M_mev=0.5 * (DD2["M_n"] + DD2["M_p"]))
    for k, tol in (("b_s", 2e-3), ("b_w", 2e-3), ("a_r", 2e-3)):
        assert abs(c[k] - DD2[k]) < tol, (k, c[k], DD2[k])
    s2 = saturation_properties(c)
    for k in ("n0", "BA", "K", "E_sym", "L"):
        assert abs(s2[k] - s[k]) < 0.01 * max(1.0, abs(s[k])), k


def test_inversion_rejects_growing_rho_coupling():
    with pytest.raises(ValueError):
        couplings_from_saturation_dd(L=110.0)          # needs a_rho < 0


def test_composition_charge_neutral():
    comp = DDRMFEOS().composition(DD2)
    x_p, x_e, x_mu = (np.asarray(a) for a in (comp.x_p, comp.x_e, comp.x_mu))
    np.testing.assert_allclose(x_e + x_mu, x_p, rtol=1e-8)
    assert np.all((x_p > 0.0) & (x_p < 0.5))


def test_ddme2_no_nucleonic_durca():
    """Raduta, Sedrakian & Weber, MNRAS 475, 4347 (2018), arXiv:1712.00584,
    Table 2: no nucleonic dUrca for DDME2.  (Fortin et al. 2016 quote
    n_DU = 0.54 fm^-3 for DD2 and DDME2; that matches the npe x_p = 1/9
    criterion, not the npe-mu triangle used here, so it is not pinned.)"""
    _, on, mdu, _ = _star(DDME2)
    assert math.isinf(mdu)


def test_mdu_falls_as_L_rises():
    """Smaller a_rho (larger L at fixed J) raises x_p at high density."""
    ms = [_star(couplings_from_saturation_dd(L=L))[2] for L in (60.0, 75.0, 90.0)]
    assert ms[2] < 2.0 and ms[2] < ms[1] <= ms[0]


def test_negative_arho_optional():
    """a_rho < 0 is allowed on request, and the saturation point is kept."""
    c = couplings_from_saturation_dd(L=110.0, allow_negative_arho=True)
    assert c["a_r"] < 0.0
    s = saturation_properties(c)
    assert abs(s["L"] - 110.0) < 0.2 and abs(s["E_sym"] - 31.7) < 0.05


def test_free_shape_parameters():
    """Non-DD2 (d_sigma, d_omega) still reproduce the requested saturation
    point and change the high-density stiffness."""
    c = couplings_from_saturation_dd(d_s=0.45, d_w=0.50)
    s = saturation_properties(c)
    for k, v in dict(n0=0.149065, BA=-16.02, K=242.7, E_sym=31.7, L=55.0).items():
        assert abs(s[k] - v) < 0.01 * max(1.0, abs(v)), k
    c0 = couplings_from_saturation_dd()
    assert abs(_star(c)[0] - _star(c0)[0]) > 0.02
