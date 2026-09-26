"""Physical sanity of the nuclear metamodel EOS."""
import numpy as np
from nmbackend.eos import MetaModelEOS
from nmbackend.constants import MEVFM3_TO_GEOM, N_SAT_DEFAULT


def test_table_monotone_and_causal():
    t = MetaModelEOS().eos_table()
    n = np.asarray(t.n)
    p = np.asarray(t.p)
    eps = np.asarray(t.eps)
    h = np.asarray(t.h)
    assert np.all(np.diff(n) > 0)
    assert np.all(np.diff(h) > 0)            # enthalpy increases with density
    assert np.all(eps > 0) and np.all(p > 0)
    assert np.all(np.asarray(t.cs2) <= 1.0 + 1e-9)   # causal


def test_saturation_pressure_small():
    # symmetric-matter pressure vanishes at saturation; beta-eq matter near n_sat
    # should still have a modest pressure (not huge, not negative)
    t = MetaModelEOS().eos_table()
    n = np.asarray(t.n)
    p_mev = np.asarray(t.p_mevfm3)
    i = int(np.argmin(np.abs(n - N_SAT_DEFAULT)))
    assert 0.0 < p_mev[i] < 20.0   # MeV/fm^3, order-of-magnitude check


def test_stiffness_knob_increases_mmax():
    from nmbackend.tov import build_family, m_max
    mm = MetaModelEOS()
    mmax_soft = float(m_max(build_family(mm.eos_table({"cs2_tail": 0.3}), n_steps=500)))
    mmax_stiff = float(m_max(build_family(mm.eos_table({"cs2_tail": 0.7}), n_steps=500)))
    assert mmax_stiff > mmax_soft


def test_bounded_symmetry_energy_above_match():
    """s_high='power*' keeps x_p below n_match and bounds S above it."""
    par = {"Q_sym": 800.0, "gamma_S": 0.5}
    tay = MetaModelEOS(muons=True).composition(par)
    for mode in ("power", "power_matched"):
        c = MetaModelEOS(muons=True, s_high=mode).composition(par)
        n, xp, xt = np.asarray(c.n), np.asarray(c.x_p), np.asarray(tay.x_p)
        n_m = 2.0 * N_SAT_DEFAULT
        lo = n <= n_m * (1 - 1e-9)
        np.testing.assert_allclose(xp[lo], xt[lo], rtol=1e-10)
        assert np.all(xp[n > 2 * n_m] < xt[n > 2 * n_m])  # cubic S is steeper
