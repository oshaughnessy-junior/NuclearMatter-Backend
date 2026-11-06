"""Independent closed-domain diffusion invariants, not stellar-cooling validation."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from nmbackend.thermal.diffusion import build_operator, implicit_step


def test_balance_equilibrium_and_signed_source():
    r = jnp.linspace(0., 1., 13)
    mu = jnp.linspace(0., 1., 9)
    shape = (12, 8)
    N = jnp.linspace(.65, .85, 12)[:, None]
    G = 1 + .15 * (1 - ((mu[1:] + mu[:-1])/2)**2)[None, :]
    op = build_operator(r, mu, N, 1.3, 1.2, G, 2., .4)
    constant = jnp.ones(shape)
    result, residual = implicit_step(op, constant, .2)
    np.testing.assert_allclose(result, constant, rtol=1e-9, atol=1e-9)
    assert float(residual) < 1e-10
    u = 1 + .2 * jnp.sin(jnp.arange(96).reshape(shape))
    v, residual = implicit_step(op, u, .2)
    np.testing.assert_allclose(jnp.sum(op.capacity*v), jnp.sum(op.capacity*u),
                               rtol=1e-10)
    assert float(jnp.sum(op.capacity*v*v)) <= float(jnp.sum(op.capacity*u*u))
    cooled, residual = implicit_step(op, constant, .1, source=-.02)
    np.testing.assert_allclose(jnp.sum(op.capacity*(cooled-constant)),
                               -.002*jnp.sum(op.source_weight), rtol=1e-8)
    assert np.max(cooled) < 1.


def test_spherical_eigenfunction_refines():
    # Independent analytic insulated-ball eigenmode: tan(kR)=kR.
    k = 4.493409457909064
    errors = []
    for nr in (16, 32, 64):
        faces = jnp.linspace(0., 1., nr+1)
        r = (faces[:-1]+faces[1:])/2
        op = build_operator(faces, jnp.linspace(0., 1., 5), 1.,1.,1.,1.,1.,1.)
        u = jnp.broadcast_to(jnp.sinc(k*r[:,None]/jnp.pi), (nr,4))
        dt = .04/nr**2
        v, residual = implicit_step(op,u,dt)
        # Compare decay rates, rather than tiny endpoint differences.
        rate = jnp.sum(op.capacity*u*(u-v))/dt/jnp.sum(op.capacity*u*u)
        errors.append(abs(float(rate/k**2-1)))
        assert float(residual) < 1e-10
    assert errors[2] < errors[1]/3 < errors[0]/9
    assert errors[-1] < .002

    # Isolate angular geometry: midpoint P2, one radial shell, Neumann faces.
    rf=jnp.array([.5,1.])
    mf=jnp.linspace(0.,1.,17)
    mu=(mf[1:]+mf[:-1])/2
    op=build_operator(rf,mf,1.,1.,1.,1.,1.,1.)
    p2=((3*mu**2-1)/2)[None,:]
    p2=p2-jnp.sum(op.capacity*p2)/jnp.sum(op.capacity)
    lam=6*.5/((1.**3-.5**3)/3)
    v,residual=implicit_step(op,p2,.03)
    np.testing.assert_allclose(v,p2/(1+.03*lam),rtol=1e-9,atol=1e-10)
    assert float(residual)<1e-10


def test_conditional_transport_gradient():
    faces = jnp.linspace(0.,1.,17)
    mu = jnp.linspace(0.,1.,9)
    u = 1 + .2*jnp.cos(jnp.arange(128).reshape(16,8))
    def endpoint(log_k):
        op=build_operator(faces,mu,.8,1.2,1.1,1.05,2.,jnp.exp(log_k))
        v,_=implicit_step(op,u,.01)
        mean=jnp.sum(op.capacity*v)/jnp.sum(op.capacity)
        return jnp.sum(op.capacity*(v-mean)**2)
    ad=float(jax.grad(endpoint)(0.))
    fd=float((endpoint(1e-4)-endpoint(-1e-4))/2e-4)
    assert ad < 0
    np.testing.assert_allclose(ad,fd,rtol=2e-5)


def test_nonuniform_interface_resistance():
    op=build_operator(jnp.array([0.,.2,1.]),jnp.array([0.,1.]),
                      1.,1.,1.,1.,1.,jnp.array([[2.],[5.]]))
    # Face at .2, distances .1 and .4, series resistance .1/2+.4/5.
    np.testing.assert_allclose(op.radial[0,0],.2**2/(.1/2+.4/5))
