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
    """Symmetric matter reproduces ALL six GM1 saturation properties, incl. the
    isovector sector (E_sym, L)."""
    from nmbackend.eos.rmft import saturation_properties, GM1
    s = saturation_properties(GM1)
    assert abs(s["n0"] - 0.153) < 0.005        # fm^-3
    assert abs(s["BA"] + 16.3) < 0.5           # MeV
    assert abs(s["K"] - 300.0) < 20.0          # MeV
    assert abs(s["mstar_frac"] - 0.70) < 0.02
    assert abs(s["E_sym"] - 32.5) < 1.0        # MeV  (isovector)
    assert abs(s["L"] - 94.0) < 8.0            # MeV  (isovector slope)


def test_beta_eq_proton_fraction_physical():
    rm = RMFTEOS()
    xp = np.array(_solve_xp(jnp.asarray(rm.n_grid), dict(DEFAULT_PARAMS)))
    assert np.all(xp > 0.0) and np.all(xp < 0.5)
    assert xp[0] < 0.1                       # neutron-rich at low density


def test_muons_switch_on_at_high_density():
    from nmbackend.eos.rmft import _leptons
    rm = RMFTEOS()
    xp = np.array(_solve_xp(jnp.asarray(rm.n_grid), dict(DEFAULT_PARAMS)))
    n = np.array(rm.n_grid)
    nmu_lo = float(_leptons(jnp.asarray(xp[20] * n[20]))[2])    # low density
    nmu_hi = float(_leptons(jnp.asarray(xp[-1] * n[-1]))[2])    # high density
    assert nmu_lo == 0.0 and nmu_hi > 1e-4                      # muons turn on


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
