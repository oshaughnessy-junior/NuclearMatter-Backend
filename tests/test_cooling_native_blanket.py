"""Independent analytic transport oracles for the smooth steady-layer helper."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nmbackend.cooling.physics.blanket import _steady_layer


def evaluate(q,cells,kind):
    luminosity=q[0]*1e35
    lam=1e20*jnp.exp(q[1])
    Tb=3e8*jnp.exp(q[2])
    rin,rout,mass=q[3:]
    radii=jnp.linspace(rout,rin,cells+1)
    if kind=='schwarzschild':
        geometry=lambda r:(jnp.sqrt(1-2*mass/r),2*mass/r)
    else:
        geometry=lambda r:(jnp.asarray(.8),jnp.asarray(0.))
    conductivity=(lambda T,r:lam*T/3e8) if kind=='linear_T' else (lambda T,r:lam)
    return _steady_layer(radii,luminosity,Tb,geometry,conductivity)


def oracle(q,kind):
    ell,loglam,logTb,rin,rout,mass=np.asarray(q)
    lam=1e20*np.exp(loglam);Tb=3e8*np.exp(logTb)
    A0=1e35/(4*np.pi*1e5*lam)
    if kind=='linear_T':
        B0=A0*3e8*(1/rin-1/rout)
        Tin=np.sqrt(Tb*Tb+2*ell*B0)
        gradient=[B0/Tin,-ell*B0/Tin,Tb*Tb/Tin,-ell*A0*3e8/rin**2/Tin,ell*A0*3e8/rout**2/Tin,0.]
    elif kind=='schwarzschild':
        ai,ao=1-2*mass/rin,1-2*mass/rout
        logratio=np.log1p(-2*mass/rout)-np.log1p(-2*mass/rin)
        D0=A0*logratio/(2*mass);D=ell*D0;Tin=Tb+D
        gradient=[D0,-D,Tb,-ell*A0/rin**2/ai,ell*A0/rout**2/ao,
                  ell*A0*(1/rin/ai-1/rout/ao)/mass-D/mass]
    else:
        D0=A0/.8*(1/rin-1/rout);D=ell*D0;Tin=Tb+D
        gradient=[D0,-D,Tb,-ell*A0/.8/rin**2,ell*A0/.8/rout**2,0.]
    return Tin,np.asarray(gradient)


@pytest.mark.parametrize('kind', ['flat','schwarzschild','linear_T'])
def test_analytic_layer_and_gradients(kind):
    q=jnp.asarray([1.,0.,0.,9.,12.,2.])
    exact,jacobian=oracle(q,kind)
    coarse=evaluate(q,64,kind)
    fine=evaluate(q,128,kind)
    assert np.all(fine.valid)
    assert abs(float(fine.T_inner)/exact-1)<abs(float(coarse.T_inner)/exact-1)/3
    assert abs(float(fine.T_inner)/exact-1)<3e-7
    function=lambda x:evaluate(x,128,kind).T_inner
    forward=np.asarray(jax.jacfwd(function)(q))
    reverse=np.asarray(jax.grad(function)(q))
    scale=np.maximum(np.abs(jacobian),1.)
    assert np.max(np.abs(forward-jacobian)/scale)<1e-5
    assert np.max(np.abs(forward-reverse)/scale)<1e-10
    direction=jnp.asarray([.2,-.1,.3,.2,-.1,.05])
    expected=float(jnp.dot(forward,direction))
    for delta in (1e-4,3e-5,1e-5):
        fd=float((function(q+delta*direction)-function(q-delta*direction))/(2*delta))
        assert abs(fd/expected-1)<1e-6
    zero=q.at[0].set(0.)
    expected,jacobian=oracle(zero,kind)
    assert abs(float(function(zero))/expected-1)<1e-12
    flux_gradient=float(jax.grad(lambda ell:function(zero.at[0].set(ell)))(0.))
    assert abs(flux_gradient/jacobian[0]-1)<1e-5


@pytest.mark.parametrize('problem', ['radius','boundary','luminosity','lapse','compactness','conductivity','bracket','local_temperature','endpoint_temperature'])
def test_invalid_layer_domain(problem):
    radii=jnp.asarray([12.,10.,9.]); L=1e35; T=3e8
    phi=.8; compactness=0.; lam=1e20
    if problem=='radius': radii=radii.at[1].set(jnp.nan)
    elif problem=='boundary': T=jnp.inf
    elif problem=='luminosity': L=jnp.nan
    elif problem=='lapse': phi=jnp.inf
    elif problem=='compactness': compactness=jnp.inf
    elif problem=='conductivity': lam=-1.
    elif problem=='local_temperature': L=0.; phi=1e-310
    elif problem=='endpoint_temperature': T=1e308; L=1e36; phi=1.; radii=jnp.asarray([12.,9.])
    else: lam=1.
    conductivity=(lambda T,r:3e28/T) if problem=='endpoint_temperature' else (lambda T,r:lam)
    result=_steady_layer(radii,L,T,lambda r:(phi,compactness),conductivity)
    assert np.isnan(float(result.T_inner))
    assert not np.all(result.valid)


def test_missing_layer_interval():
    with pytest.raises(ValueError,match='at least two'):
        _steady_layer(jnp.asarray([10.]),0.,3e8,lambda r:(.8,0.),lambda T,r:1e20)
