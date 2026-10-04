"""Smooth-branch steady conduction in prescribed spherical geometry."""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp

_K = 4*jnp.pi*1e5  # km radius and cm conductivity units


class _LayerSolution(NamedTuple):
    T_inner: jax.Array
    residual: jax.Array
    slope: jax.Array
    conductivity: jax.Array
    valid: jax.Array


def _steady_layer(radii, luminosity, boundary_T, geometry, conductivity):
    """Steady constant-flux layer with descending radii in km and redshifted T.

    Geometry returns (lapse, 2m/r); conductivity takes local T and radius,
    returning erg cm^-1 s^-1 K^-1. Requires a smooth conductivity branch and
    unique non-tangent step roots. No storage, heating or neutrino sinks.
    Invalid numerical domains return NaN temperature and false step validity.
    """
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError('Layer requires at least two one-dimensional radii')
    def step(yout, shell):
        rout,rin=shell
        r=.5*(rin+rout)
        phi,compactness=geometry(r)
        scale=(rout-rin)*luminosity/(_K*r*r*phi*jnp.sqrt(1-compactness))
        def residual(yin):
            temperature=jnp.exp(.5*(yin+yout))
            lam=conductivity(temperature/phi,r)
            return yin-yout-scale/(lam*temperature)
        def solve(fun,initial):
            def expand(_,hi):
                return jax.lax.cond(fun(hi)<0,lambda x:yout+2*(x-yout),lambda x:x,hi)
            hi=jax.lax.fori_loop(0,8,expand,yout+.05)
            def bisect(_,bounds):
                lo,hi=bounds
                mid=.5*(lo+hi)
                above=fun(mid)>0
                return jnp.where(above,lo,mid),jnp.where(above,mid,hi)
            lo,upper=jax.lax.fori_loop(0,48,bisect,(yout,hi))
            root=jnp.where(fun(initial)==0,initial,.5*(lo+upper))
            supported=jnp.isfinite(fun(hi))&(fun(hi)>=0)&jnp.isfinite(fun(yout))&(fun(yout)<=0)
            return jnp.where(supported,root,jnp.nan)
        root=jax.lax.custom_root(residual,yout,solve,lambda g,y:y/g(jnp.ones_like(y)))
        slope=jax.grad(residual)(root)
        local_T=jnp.exp(.5*(root+yout))/phi
        inner_T=jnp.exp(root)
        lam=conductivity(local_T,r)
        error=jnp.abs(residual(root))
        valid=(jnp.isfinite(root)&jnp.isfinite(local_T)&(local_T>0)
               &jnp.isfinite(inner_T)&(inner_T>0)&jnp.isfinite(error)&(error<3e-12)&jnp.isfinite(slope)&(slope>1e-8)
               &jnp.isfinite(lam)&(lam>0)&jnp.isfinite(rout)&jnp.isfinite(rin)&(rout>rin)&(rin>0)
               &jnp.isfinite(phi)&(phi>0)&jnp.isfinite(compactness)&(compactness>=0)&(compactness<1))
        return jnp.where(valid,root,jnp.nan),(error,slope,lam,valid)
    y,diagnostic=jax.lax.scan(step,jnp.log(boundary_T),(radii[:-1],radii[1:]))
    valid=(jnp.all(jnp.isfinite(radii))&jnp.all(jnp.diff(radii)<0)
           &jnp.isfinite(luminosity)&(luminosity>=0)&jnp.isfinite(boundary_T)&(boundary_T>0))
    return _LayerSolution(jnp.where(valid,jnp.exp(y),jnp.nan),
                          *diagnostic[:3], diagnostic[3]&valid)
