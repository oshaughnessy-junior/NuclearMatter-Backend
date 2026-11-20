"""Superfluid gap and reduction-factor fits used by NSCool.

u(tau) is the dimensionless gap v = Delta / T at tau = T / Tc
(Levenfish & Yakovlev 1994).  ``below`` returns tau = 0.5 outside the
paired region and tau <= 1 - 1e-12 inside it, so that the derivative of
sqrt(1 - tau), and of every branch a ``jnp.where`` discards, stays finite.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["below", "u_1s0", "u_3p2b", "fexp"]


def fexp(x):
    """NSCool's fexp: exp(max(x, -700))."""
    return jnp.exp(jnp.maximum(x, -700.0))


def below(t, tc, strict=True):
    """(mask, tau): mask = T < Tc (or <=); tau = T/Tc there (at most
    1 - 1e-12), else 0.5.  Tc <= 0 counts as unpaired."""
    m = ((t < tc) if strict else (t <= tc)) & (tc > 0)
    tcs = jnp.where(m, tc, 1.0)
    return m, jnp.where(m, jnp.minimum(t / tcs, 1.0 - 1e-12), 0.5)


def u_1s0(tau):
    """Singlet gap v(tau) (NSCool u_1s0)."""
    return jnp.sqrt(1.0 - tau) * (1.456 - 0.157 / jnp.sqrt(tau) + 1.764 / tau)


def u_3p2b(tau):
    """Triplet m=0 gap v(tau) (NSCool u_3p2B in the Urca and brem routines)."""
    return jnp.sqrt(1.0 - tau) * (0.7893 + 1.188 / tau)


def f32(x):
    """A single-precision Fortran literal (no d0), widened as gfortran does."""
    return float(np.float32(x))
