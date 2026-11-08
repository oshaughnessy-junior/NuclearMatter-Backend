"""Relativistic polytropes in K=G=c=1, with analytic enthalpy inversion."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def eos_at_enthalpy(h, index=1.):
    gamma = 1 + 1/index
    # A fractional power at zero has an infinite inactive derivative. Keep its
    # evaluated branch positive before selecting exact vacuum support.
    positive = jnp.expm1(jnp.maximum(h, 1e-12))*(gamma-1)/gamma
    rho0 = jnp.where(h > 0, positive**index, 0.)
    p = rho0**gamma
    return rho0 + p/(gamma-1), p, rho0


@jax.custom_jvp
def _central_rest_density(energy, index):
    """Primal bisection with an implicit derivative, not branch-history AD."""
    gamma = 1 + 1/index
    def step(_, bounds):
        lo, hi = bounds
        mid = (lo+hi)/2
        above = mid + mid**gamma/(gamma-1) > energy
        return jnp.where(above, lo, mid), jnp.where(above, mid, hi)
    lo, hi = jax.lax.fori_loop(0, 64, step, (jnp.zeros_like(energy), energy))
    return (lo+hi)/2


@_central_rest_density.defjvp
def _rest_jvp(primals, tangents):
    energy,index=primals
    de,di=tangents
    rho=_central_rest_density(energy,index)
    gamma=1+1/index
    p=rho**gamma
    density_slope=1+index*gamma*rho**(gamma-1)
    index_slope=p*(1-jnp.log(rho)/index)
    return rho,(de-index_slope*di)/density_slope


def central_enthalpy(energy, index=1.):
    """Invert central energy, with the converged EOS root differentiated implicitly."""
    gamma=1+1/index
    rho0=_central_rest_density(energy,index)
    return jnp.log1p(gamma/(gamma-1)*rho0**(gamma-1))


def branch_input(energy, q, eos_points=16, index=1.):
    hc = central_enthalpy(energy, index)
    h = hc*jnp.linspace(0., 1., eos_points)
    eps, p, _ = eos_at_enthalpy(h, index)
    return jnp.concatenate([eps/energy, p/energy, jnp.array([hc, q])])
