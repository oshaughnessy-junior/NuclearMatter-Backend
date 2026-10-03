"""Crust junction derivatives against algebra, chemistry against NSCool dumps."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.cooling import NSCool, default_home
from nmbackend.cooling.physics.crust import CrustTable, crust_chemistry


def table():
    return CrustTable(*(jnp.array(v,dtype=float) for v in
                        ([1.,2.,4.],[.01,.02,.04],[10.,10.,30.],
                         [10.,10.,10.],[3.,4.,5.])))


def evaluate(x):
    rho,core_rho,core_n,core_ye=x
    c=crust_chemistry(table(),rho,core_rho=core_rho,core_n=core_n,core_Ye=core_ye)
    return jnp.array([c.n,c.A_cell,c.Ye,c.Yn])


def test_virtual_join_matches_analytic_moving_endpoint():
    x=jnp.array([5.,6.,.06,.1])
    np.testing.assert_allclose(jax.jit(evaluate)(x),[.05,40.,.125,.75],rtol=1e-14)
    expected=np.array([[.01,-.005,.5,0.],[10.,-5.,0.,-250.],
                       [-.03125,.015625,0.,.78125],[.0625,-.03125,0.,-1.5625]])
    forward=np.asarray(jax.jacfwd(evaluate)(x))
    np.testing.assert_allclose(forward,expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(jax.jacrev(evaluate)(x),expected,rtol=1e-13,atol=1e-13)
    for step in (1e-5,3e-6,1e-6):
        for k in range(4):
            offset=jnp.zeros(4).at[k].set(step)
            fd=np.asarray((evaluate(x+offset)-evaluate(x-offset))/(2*step))
            np.testing.assert_allclose(fd,expected[:,k],rtol=3e-8,atol=1e-7)


def test_species_contacts_use_left_interval_and_outer_gradients_are_finite():
    c=crust_chemistry(table(),2.,core_rho=6.,core_n=.06,core_Ye=.1)
    np.testing.assert_allclose([c.A_cell,c.A_ion,c.Z,c.Ye,c.Yn],[10.,10.,4.,.4,0.])
    c=crust_chemistry(table(),2.0001,core_rho=6.,core_n=.06,core_Ye=.1)
    assert float(c.Z)==5. and float(c.Yn)>0.
    jac=jax.jacfwd(evaluate)(jnp.array([1.5,6.,.06,.1]))
    assert np.all(np.isfinite(jac))
    np.testing.assert_allclose(jac[1:],0.,atol=0.)
    np.testing.assert_allclose(jac[0],[.01,0.,0.,0.],rtol=1e-14)


def test_invalid_queries_and_joins_are_flagged():
    for x in ([.9,6.,.06,.1],[6.1,6.,.06,.1],[5.,6.,.06,0.],
              [1.5,6.,.06,.9],[1.5,6.,.01,.1],[5.,6.,.06,1e-320]):
        assert np.all(np.isnan(evaluate(jnp.array(x))))


@pytest.mark.skipif(not (default_home()/'Code/NSCool_iso.out').is_file()
                    and not os.environ.get('NSCOOL_REQUIRED'),reason='NSCool not built')
@pytest.mark.parametrize('mass',[1.4,2.])
def test_chemistry_matches_real_crust_and_virtual_join(mass):
    from nmbackend.cooling.physics.reference import dump
    d=dump(mass,'none')
    crust=d.zones[d.zones>d.icore]
    s=d.star
    c=jax.jit(crust_chemistry)(CrustTable.read(NSCool().default_crust),
                              jnp.asarray(s['rho'][crust]),core_rho=s['rho'][d.icore],
                              core_n=s['nb'][d.icore],core_Ye=s['Ye'][d.icore])
    for ours,theirs in [('n','nb'),('Ye','Ye'),('Yn','Yn'),('A_cell','A_cell'),
                        ('A_ion','A_ion'),('Z','Z_ion'),('v_ion','v_ion')]:
        np.testing.assert_allclose(getattr(c,ours),s[theirs][crust],rtol=3e-7,atol=1e-10,err_msg=ours)
    assert np.any(s['A_cell'][crust]==s['A_ion'][crust])
    assert np.any(s['A_cell'][crust]>s['A_ion'][crust])
    assert np.all(np.isfinite(np.asarray(c)))
