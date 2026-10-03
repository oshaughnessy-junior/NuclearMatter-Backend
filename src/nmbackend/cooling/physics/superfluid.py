"""Superfluid gap and reduction-factor fits used by NSCool.

u(tau) is the dimensionless gap v = Delta / T at tau = T / Tc
(Levenfish & Yakovlev 1994).  NSCool's functions are evaluated with
``tau`` clipped below 1 so that gradients of the unused branch of a
``jnp.where`` stay finite; callers select tau < 1 themselves.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["below", "u_1s0", "u_3p2b", "fexp"]


def fexp(x):
    """NSCool's fexp: exp(max(x, -700))."""
    return jnp.exp(jnp.maximum(x, -700.0))


def below(t, tc, strict=True):
    """(mask, tau) with mask = T < Tc (or <=), tau = T/Tc where masked, else 0.5."""
    m = (t < tc) if strict else (t <= tc)
    return m, jnp.where(m, t / tc, 0.5)


def u_1s0(tau):
    """Singlet gap v(tau) (NSCool u_1s0)."""
    return jnp.sqrt(1.0 - tau) * (1.456 - 0.157 / jnp.sqrt(tau) + 1.764 / tau)


def u_3p2b(tau):
    """Triplet m=0 gap v(tau) (NSCool u_3p2B in the Urca and brem routines)."""
    return jnp.sqrt(1.0 - tau) * (0.7893 + 1.188 / tau)


def f32(x):
    """A single-precision Fortran literal (no d0), widened as gfortran does."""
    return float(np.float32(x))
