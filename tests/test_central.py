"""Tests for tov.central (family central properties).

Needs a TOV solve (JAX): run with the normal suite, not while another JAX
process occupies the login-node thread budget (ulimit -u 500 allows one).
"""
import numpy as np
import pytest

from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, family_central, central_at_mass


@pytest.fixture(scope="module")
def mm_family():
    mm = MetaModelEOS()
    params = dict(K_sat=230.0, E_sym=32.0, L_sym=60.0, cs2_tail=0.4)
    table = mm.eos_table(params)
    fam = build_family(table, n_steps=600)
    return table, fam


def test_central_family_consistency(mm_family):
    table, fam = mm_family
    cf = family_central(table, fam)
    s = cf.stable
    assert s.sum() > 10
    # X positive, below causal-average unity, increasing with density
    assert np.all(cf.X[s] > 0) and np.all(cf.X[s] < 1)
    hc = cf.h_c[s]
    assert np.corrcoef(hc, cf.X[s])[0, 1] > 0.9
    # cs2_c matches the table's tail value once the center is in the tail
    assert cf.cs2_c[s].max() == pytest.approx(0.4, abs=0.02)


def test_central_at_mass_canonical(mm_family):
    table, fam = mm_family
    cf = family_central(table, fam)
    c = central_at_mass(cf, 1.4)
    # canonical-star ballparks on a physical metamodel EOS
    assert 0.08 < c["X"] < 0.25
    assert 0.5 < c["Psi"] < 10.0
    assert c["Lambda"] > 50.0
    # Psi from AD/FD should be smooth: nearby masses give nearby Psi
    c2 = central_at_mass(cf, 1.41)
    assert c2["Psi"] == pytest.approx(c["Psi"], rel=0.15)


def test_central_at_mass_out_of_range(mm_family):
    table, fam = mm_family
    cf = family_central(table, fam)
    with pytest.raises(ValueError):
        central_at_mass(cf, 5.0)
