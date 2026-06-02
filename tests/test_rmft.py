"""Self-consistent sigma-omega-rho RMFT EOS, validated against the GM1 set."""
import numpy as np
import jax
import jax.numpy as jnp

from nmbackend.eos import RMFTEOS
from nmbackend.eos.rmft import _eps_p_of_n, _solve_phi, _solve_xp, DEFAULT_PARAMS
from nmbackend.constants import HBARC
from nmbackend.tov.family import (
    build_family, default_hc_grid, m_max, radius_of_mass,
)

M_N = 939.0


def test_gm1_saturation_properties():
    """Symmetric nuclear matter reproduces GM1 saturation (n0, B/A, m*/M, K)."""
    p = dict(DEFAULT_PARAMS)
    sym = jax.jit(lambda n: _eps_p_of_n(n, 0.5, p, leptons=False)[:2])
    ns = jnp.linspace(0.13, 0.18, 51)
    eps, pres = jax.vmap(sym)(ns)
    eps, pres, ns = np.array(eps), np.array(pres), np.array(ns)
    i0 = int(np.argmin(np.abs(pres)))            # p = 0 -> saturation
    n0 = ns[i0]
    BA = eps[i0] / n0 - M_N
    Phi = float(_solve_phi(jnp.asarray(n0 / 2), jnp.asarray(n0 / 2), p))
    mstar_frac = (M_N / HBARC - Phi) * HBARC / M_N
    K = 9.0 * (pres[i0 + 1] - pres[i0 - 1]) / (ns[i0 + 1] - ns[i0 - 1])

    assert abs(n0 - 0.153) < 0.005          # fm^-3
    assert abs(BA + 16.3) < 0.5             # MeV
    assert abs(mstar_frac - 0.70) < 0.02
    assert abs(K - 300.0) < 20.0            # MeV


def test_beta_eq_proton_fraction_physical():
    rm = RMFTEOS()
    xp = np.array(_solve_xp(jnp.asarray(rm.n_grid), dict(DEFAULT_PARAMS)))
    assert np.all(xp > 0.0) and np.all(xp < 0.5)
    assert xp[0] < 0.1                       # neutron-rich at low density


def test_eos_table_monotone_causal():
    t = RMFTEOS().eos_table()
    assert np.all(np.diff(np.asarray(t.n)) > 0)
    assert np.all(np.diff(np.asarray(t.h)) > 0)
    assert np.all(np.asarray(t.p) > 0) and np.all(np.asarray(t.eps) > 0)
    assert np.all(np.asarray(t.cs2) <= 1.0 + 1e-9)


def test_gm1_neutron_star_in_literature_ballpark():
    fam = build_family(RMFTEOS().eos_table(),
                       default_hc_grid(RMFTEOS().eos_table(), n_models=60))
    mmax = float(m_max(fam))
    r14 = float(radius_of_mass(fam, 1.4))
    assert 2.2 < mmax < 2.5                  # GM1 ~ 2.36
    assert 12.5 < r14 < 14.5                 # GM1 ~ 13.7


def test_ad_through_self_consistent_solve_finite():
    rm = RMFTEOS()
    HC = 0.30 * float(rm.eos_table().h[-1])
    def Mof(Co):
        return solve_M(rm, Co, HC)
    g = float(jax.jacfwd(Mof)(7.148))
    assert np.isfinite(g) and g > 0.0        # stiffer omega coupling -> heavier star


def solve_M(rm, Co, HC):
    from nmbackend.tov.solver import solve_structure
    return solve_structure(rm.eos_table({"C_omega": Co}), HC,
                           rtol=1e-8, atol=1e-10).M_msun
