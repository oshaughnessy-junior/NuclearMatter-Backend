"""Unpaired assembled crust rates against independent NSCool zone budgets."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nmbackend.cooling import NSCool, default_home
from nmbackend.cooling.physics.crust import (CrustTable, crust_chemistry,
                                            _unpaired_crust_terms)

pytestmark = pytest.mark.skipif(
    not (default_home()/'Code/NSCool_iso.out').is_file()
    and not os.environ.get('NSCOOL_REQUIRED'), reason='NSCool not built')


@pytest.mark.parametrize('mass', [1.4, 2.])
def test_unpaired_crust_budget_matches_reference(mass):
    from nmbackend.cooling.physics.reference import dump
    d = dump(mass, 'none')
    zones = d.zones[d.zones > d.icore]
    s = d.star
    rho = jnp.asarray(s['rho'][zones])
    chemistry = crust_chemistry(CrustTable.read(NSCool().default_crust), rho,
        core_rho=s['rho'][d.icore], core_n=s['nb'][d.icore], core_Ye=s['Ye'][d.icore])
    cv, q = jax.jit(lambda T: _unpaired_crust_terms(T, rho, chemistry,
                                                  rho_drip=d.rhodrip))(
        d.T[:, None]/s['ephi'][zones])
    indices = np.flatnonzero(d.zones > d.icore)
    np.testing.assert_allclose(cv, d.terms['cv'][:, indices], rtol=1e-6)
    np.testing.assert_allclose(q, d.terms['qnu'][:, indices], rtol=1e-6, atol=1e-300)


def test_custom_base_excludes_lower_ionic_threshold():
    """A source fit threshold below rho_base must not extend the domain."""
    from nmbackend.constants import MEVFM3_TO_GEOM
    from nmbackend.cooling.physics.geometry import enthalpy_at_energy, volume_at_h, volume_quadrature
    from nmbackend.cooling.physics.whole import _unpaired_sums
    from nmbackend.eos import MetaModelEOS
    from nmbackend.eos.coolcomp import cooling_composition
    from nmbackend.eos.crust import _G_CM3_TO_MEVFM3
    from nmbackend.tov import solve_profile
    model = MetaModelEOS(muons=True, s_high='power')
    params = {'L_sym': 100., 'K_sym': -100.}
    table = model.eos_table(params)
    comp = cooling_composition(model, params)
    hc = .3
    profile = solve_profile(table, hc, n_points=512, rtol=1e-10, atol=1e-12)
    crust = CrustTable.read(NSCool().default_crust)
    sums = _unpaired_sums(table, profile, hc, comp, crust, 3e8,
                          rho_base=1e12, rho_drip=2e12, order=16)
    hbase = enthalpy_at_energy(table, 1e12*_G_CM3_TO_MEVFM3*MEVFM3_TO_GEOM)
    # Split the independent geometric budget at every saved profile node.
    edges = jnp.sort(volume_at_h(profile, hc, jnp.clip(
        jnp.concatenate((jnp.asarray([hc, hbase]), profile.h)), hbase, hc)))
    _, weights = volume_quadrature(profile, hc, edges, order=32)
    np.testing.assert_allclose(sums.volume_cm3, jnp.sum(weights), rtol=1e-8)
