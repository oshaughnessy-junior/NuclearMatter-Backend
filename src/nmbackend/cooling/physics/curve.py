"""Isothermal cooling curve from the JAX sums (``nmbackend.cooling.isothermal``
with jnp, so it is differentiable).

t(T) = int C / (L_nu + L_gamma) dT, trapezoid in ln T on a decreasing grid
of T_b, with the crust blanket: the interior at T_in(T_b) has C(T_in) and
L_nu(T_in), interpolated log-log on the grid as the C1 driver does.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..isothermal import YEAR_S, IsothermalCurve
from .star import isothermal_sums

__all__ = ["isothermal_curve"]


def isothermal_curve(T_grid, star, tc, eta=0.0, rho_b=0.0, ca_n3p2=0.5,
                     blanket=None) -> IsothermalCurve:
    """Curve on ``T_grid`` (redshifted T_b, any order; returned decreasing).

    ``blanket`` defaults to rho_b > 0.  Heating is off (L_heat = 0).
    """
    blanket = rho_b > 0 if blanket is None else blanket
    T = jnp.sort(jnp.asarray(T_grid, float))[::-1]
    s = jax.vmap(lambda t: isothermal_sums(t, star, tc, eta, rho_b, ca_n3p2))(T)
    lnT = jnp.log(T)
    C, L_nu = s.C, s.L_nu
    if blanket:
        lnTi = jnp.log(s.T_in)
        li = lambda y: jnp.exp(jnp.interp(-lnTi, -lnT, jnp.log(jnp.maximum(y, 1e-300))))
        C, L_nu = li(C), li(L_nu)
        lnT = lnTi
    f = C / (L_nu + s.L_gamma) * jnp.exp(lnT)
    t = jnp.cumsum(0.5 * (f[1:] + f[:-1]) * -jnp.diff(lnT))
    return IsothermalCurve(t / YEAR_S, jnp.exp(lnT[1:]), s.Teff_inf[1:],
                           s.L_gamma[1:], L_nu[1:])
