"""Analytic Green inversions, EOS identities and operator symmetry/AD checks."""
from __future__ import annotations
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from nmbackend.rotating.pino import network
from nmbackend.rotating.pino.keh import Fields, make_grid, green_map,fields_from_evaluator,residual_and_observables
from nmbackend.rotating.pino.polytrope import eos_at_enthalpy,central_enthalpy,branch_input


def test_enthalpy_inverse_and_first_law():
    for index in (.8,1.,1.2):
        h=central_enthalpy(jnp.asarray(.2),index)
        e,p,rho=eos_at_enthalpy(h,index)
        np.testing.assert_allclose(e,.2,rtol=1e-13)
        dpdh=jax.grad(lambda hh:eos_at_enthalpy(hh,index)[1])(h)
        np.testing.assert_allclose(dpdh,e+p,rtol=1e-13)
        assert rho>0
        assert np.isfinite(jax.grad(lambda hh:eos_at_enthalpy(hh,index)[0])(0.))
        implicit=jax.grad(lambda n:central_enthalpy(jnp.asarray(.2),n))(index)
        finite=(central_enthalpy(jnp.asarray(.2),index+1e-5)-central_enthalpy(jnp.asarray(.2),index-1e-5))/2e-5
        np.testing.assert_allclose(implicit,finite,rtol=1e-8)


def test_radial_green_inverse_uniform_balls():
    # Independent exact D_d u=-1 inside r<=1, vacuum outside, regular/decaying.
    grid=make_grid(128,24,6)
    r=grid.s/(1-grid.s)
    zero=jnp.zeros((len(r),len(grid.mu),3))
    boundary=jnp.zeros((len(r),3))
    point=jnp.zeros(3)
    fields=Fields(zero,zero,zero,zero,zero,zero,boundary,point,point,point,1.,0.,.1,0.)
    source=jnp.broadcast_to(jnp.where(r[:,None]<=1.,-1.,0.),(len(r),len(grid.mu)))
    mapped=green_map(jnp.stack([source]*3),fields,grid)
    for component,d in enumerate((4,3,5)):
        exact=jnp.where(r<=1.,1/(2*(d-2))-r*r/(2*d),1/(d*(d-2)*r**(d-2)))
        error=np.max(np.abs(np.asarray(mapped[...,component]-exact[:,None])))
        assert error<.002


def test_architecture_symmetry_and_nonvacuous_derivative():
    params=network.initialize(jax.random.PRNGKey(42),width=8,modes=4)
    z=branch_input(jnp.asarray(.2),jnp.asarray(.8))
    np.testing.assert_allclose(network.evaluate(params,z,jnp.array([.4,.3])),network.evaluate(params,z,jnp.array([.4,-.3])),atol=0)
    static=z.at[-1].set(1.)
    assert network.evaluate(params,static,jnp.array([.4,.3]))[2]==0
    np.testing.assert_allclose(network.evaluate(params,z,jnp.array([1.,.3])),0.,atol=0)
    np.testing.assert_allclose(network.evaluate(params,z,jnp.array([0.,.3])),network.evaluate(params,z,jnp.array([0.,.7])),atol=1e-16)
    np.testing.assert_allclose(jax.jacfwd(lambda pt:network.evaluate(params,z,pt))(jnp.array([0.,.3])),0.,atol=1e-15)
    def value(scale):
        adjusted={**params,'scale':params['scale'].at[1].set(scale)}
        return network.evaluate(adjusted,z,jnp.array([.4,.3]))[1]
    ad=jax.grad(value)(.01)
    fd=(value(.010001)-value(.009999))/.000002
    assert abs(ad)>1e-6
    np.testing.assert_allclose(ad,fd,rtol=1e-7)


def test_complete_physics_loss_directional_derivative():
    params=network.initialize(jax.random.PRNGKey(43),width=8,modes=4)
    z=branch_input(jnp.asarray(.2),jnp.asarray(.8))
    grid=make_grid(16,6,3)
    def loss(offset):
        candidate={**params,'offset':params['offset'].at[1].set(offset)}
        fields=fields_from_evaluator(lambda pt:network.evaluate(candidate,z,pt),z[-2],z[-1],grid)
        return residual_and_observables(fields,grid,eos_at_enthalpy)[0]
    value,ad=jax.jit(jax.value_and_grad(loss))(-1.)
    assert np.isfinite(value) and np.isfinite(ad) and abs(ad)>1e-6
    for h in (1e-4,1e-5):
        fd=(loss(-1+h)-loss(-1-h))/(2*h)
        np.testing.assert_allclose(ad,fd,rtol=2e-4)
