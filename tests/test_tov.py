"""TOV solver sanity: realistic NS, branch turnover, causality, tidal."""
import numpy as np
from nmbackend.eos import MetaModelEOS, PolytropeEOS
from nmbackend.tov import (
    build_family, m_max, lambda_of_mass, radius_of_mass, solve_structure,
)


def test_metamodel_star_is_physical():
    fam = build_family(MetaModelEOS().eos_table(), n_steps=700)
    mmax = float(m_max(fam))
    r14 = float(radius_of_mass(fam, 1.4))
    l14 = float(lambda_of_mass(fam, 1.4))
    assert 1.9 < mmax < 2.6          # supports a heavy NS
    assert 10.0 < r14 < 14.0         # km
    assert 100.0 < l14 < 1200.0      # GW170817-ballpark


def test_branch_turnover_and_compactness():
    fam = build_family(MetaModelEOS().eos_table(), n_steps=700)
    stable = np.asarray(fam.stable)
    # one stable block (minimum-mass star to M_max), then unstable past M_max;
    # with the crust the lowest grid stars can lie below the minimum mass
    i = np.flatnonzero(stable)
    assert i.size > 10 and np.all(np.diff(i) == 1) and (not stable[-1])
    C = np.asarray(fam.C)
    assert np.all(C < 0.5)           # below the BH/Buchdahl-ish limit


def test_lambda_decreases_with_mass():
    fam = build_family(MetaModelEOS().eos_table(), n_steps=700)
    masses = np.linspace(1.0, 1.8, 5)
    L = np.array([float(lambda_of_mass(fam, m)) for m in masses])
    assert np.all(np.diff(L) < 0)    # Lambda falls as mass rises


def test_polytrope_runs():
    fam = build_family(PolytropeEOS().eos_table(), n_steps=500)
    assert float(m_max(fam)) > 0.8
    pt = solve_structure(PolytropeEOS().eos_table(), 0.3, n_steps=400)
    assert float(pt.R_km) > 0 and float(pt.M_msun) > 0
