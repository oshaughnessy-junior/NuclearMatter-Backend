"""Differentiable TOV solver in pseudo-enthalpy, with in-loop tidal Love number.

We integrate the TOV equations using the pseudo-enthalpy ``h`` as the independent
variable (Lindblom 1992), which removes the surface coordinate singularity and
makes the central boundary condition a simple series start.  Alongside (r, m) we
carry the tidal variable y (Postnikov, Prakash & Lattimer 2010) to obtain the
l=2 Love number k2 and dimensionless tidal deformability Lambda.

The integrator is a fixed-step RK4 implemented with ``jax.lax.scan`` so the whole
solve is ``jit``/``vmap``/``grad``-able: forward-mode AD through this loop yields
exactly the perturbation ("delta-TOV") sensitivities one would otherwise derive by
hand (cf. Z. Zhu ``delta_tt`` -- motivated-by; see ATTRIBUTION.md).

Geometric units throughout: r, m in km; eps, p in km^-2.  This module exposes a
single ``backend`` seam (``solve_structure``) so an alternative high-accuracy or
compiled backend (RePrimAnd; Zhu C++ delta-TOV) can be slotted in later without
changing callers.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..constants import GMSUN_KM
from ..eos.base import EOSTable

jax.config.update("jax_enable_x64", True)

_FOURPI = 4.0 * math.pi


class TOVPoint(NamedTuple):
    """One stellar model."""
    h_c: jnp.ndarray      # central pseudo-enthalpy
    R_km: jnp.ndarray     # circumferential radius, km
    M_km: jnp.ndarray     # gravitational mass, geometric km
    M_msun: jnp.ndarray   # gravitational mass, solar masses
    C: jnp.ndarray        # compactness M/R (geometric)
    y: jnp.ndarray        # surface tidal variable
    k2: jnp.ndarray       # l=2 tidal Love number
    Lambda: jnp.ndarray   # dimensionless tidal deformability


def _derivs(h, state, table: EOSTable, tidal: bool = True):
    r, m, y = state
    eps, p, cs2 = table.at_h(h)
    denom = m + _FOURPI * r**3 * p
    drdh = -r * (r - 2.0 * m) / denom
    dmdh = _FOURPI * r**2 * eps * drdh

    if tidal:
        one_minus_2mr = 1.0 - 2.0 * m / r
        F = (1.0 - _FOURPI * r**2 * (eps - p)) / one_minus_2mr
        Q = (_FOURPI / one_minus_2mr) * (
            5.0 * eps + 9.0 * p + (eps + p) / cs2 - 6.0 / (_FOURPI * r**2)
        ) - (2.0 * (m + _FOURPI * r**3 * p) / (r**2 * one_minus_2mr)) ** 2
        dydr = -(y**2 + y * F + r**2 * Q) / r
        dydh = dydr * drdh
    else:
        # structure-only: keep y constant so its singular near-center terms are
        # never formed -> reverse-mode AD of (R, M) is clean.
        dydh = 0.0 * r
    return jnp.stack([drdh, dmdh, dydh])


def _k2_from_y_C(y, C):
    """l=2 tidal Love number from surface y and compactness C (Hinderer 2008)."""
    onemtwoC = 1.0 - 2.0 * C
    num = (8.0 / 5.0) * C**5 * onemtwoC**2 * (2.0 + 2.0 * C * (y - 1.0) - y)
    den = (
        2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
        + 4.0 * C**3 * (13.0 - 11.0 * y + C * (3.0 * y - 2.0)
                        + 2.0 * C**2 * (1.0 + y))
        + 3.0 * onemtwoC**2 * (2.0 - y + 2.0 * C * (y - 1.0)) * jnp.log(onemtwoC)
    )
    return num / den


def solve_structure(table: EOSTable, h_c, n_steps: int = 800, tidal: bool = True):
    """Integrate one star at central pseudo-enthalpy ``h_c``.  Returns TOVPoint.

    This is the *backend seam*: callers (family, adapters) depend only on this
    signature and the TOVPoint output, not on the integration internals.

    Gradient guidance: prefer **forward-mode** AD (``jax.jacfwd`` / ``jax.jvp``)
    for sensitivities of any TOV output -- it is clean, efficient for our small
    parameter count, and reproduces the perturbation ("delta-TOV") equations.
    Reverse-mode (``jax.grad``) is supported for the **structure-only** outputs
    (R, M) via ``tidal=False``; reverse-mode through the tidal variable is not
    recommended (the near-center 1/r^2 terms give 0*inf in the backward pass).
    """
    h_min = table.h[0]
    h_c = jnp.asarray(h_c)
    dh0 = (h_c - h_min) * 1e-6
    h_start = h_c - dh0

    eps_c, p_c, _ = table.at_h(h_c)
    # leading-order central series start
    r0 = jnp.sqrt(3.0 * dh0 / (2.0 * math.pi * (eps_c + 3.0 * p_c)))
    m0 = (4.0 / 3.0) * math.pi * eps_c * r0**3
    state0 = jnp.stack([r0, m0, jnp.asarray(2.0)])

    hs = jnp.linspace(h_start, h_min, n_steps + 1)
    h_lefts = hs[:-1]
    dhs = jnp.diff(hs)

    def step(state, hd):
        h0, dh = hd
        k1 = _derivs(h0, state, table, tidal)
        k2 = _derivs(h0 + 0.5 * dh, state + 0.5 * dh * k1, table, tidal)
        k3 = _derivs(h0 + 0.5 * dh, state + 0.5 * dh * k2, table, tidal)
        k4 = _derivs(h0 + dh, state + dh * k3, table, tidal)
        state = state + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state, None

    state, _ = jax.lax.scan(step, state0, (h_lefts, dhs))
    R, M, y = state[0], state[1], state[2]
    C = M / R
    if tidal:
        k2 = _k2_from_y_C(y, C)
        Lam = (2.0 / 3.0) * k2 / C**5
    else:
        k2 = jnp.nan * R
        Lam = jnp.nan * R
    return TOVPoint(
        h_c=h_c, R_km=R, M_km=M, M_msun=M / GMSUN_KM,
        C=C, y=y, k2=k2, Lambda=Lam,
    )


# Convenience: jitted single solve (n_steps fixed as a Python int -> static)
def make_solver(n_steps: int = 800):
    @jax.jit
    def _solve(table: EOSTable, h_c):
        return solve_structure(table, h_c, n_steps=n_steps)
    return _solve
