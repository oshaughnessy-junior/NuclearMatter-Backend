"""beta-equilibrium composition and Landau effective masses for cooling.

``cooling_composition(model, params)`` returns, on the model's composition
grid, the particle fractions per baryon and the Landau effective masses at
the Fermi surface (units of the bare nucleon mass).  Prescriptions:

  * Skyrme (sk, skx): exact for the functional, h2m/m*_q = h2m + c1 n + c2 n_q.
  * RMF (rmf, rmfwr) and DD-RMF (dd, ddf): m*_q = E*_Fq = sqrt(kF_q^2 + m_D^2).
  * metamodel (mm): Margueron+2018 form with m*_sat/m = 0.75,
    Delta m*_sat/m = 0.10 (see :mod:`nmbackend.eos.metamodel`).

``pack_d0`` / ``unpack_d0`` store (x_p, x_mu, m*_n, m*_p) of one draw on the
fixed grid ``D0_GRID`` (the paper-6 x_p grid), float32, NaN above the model's
grid.  x_n = 1 - x_p and x_e = x_p - x_mu are not stored.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

__all__ = ["CoolingComposition", "cooling_composition", "D0_GRID",
           "D0_FIELDS", "pack_d0", "unpack_d0"]

#: fm^-3; paper 6 c1_pilot2 N_GRID
D0_GRID = np.linspace(0.08, 1.2, 57)
D0_FIELDS = ("x_p", "x_mu", "mst_n", "mst_p")


class CoolingComposition(NamedTuple):
    """Fractions per baryon and Landau masses m*/m on a density grid (fm^-3).

    Has the fields of :class:`nmbackend.eos.Composition`, so
    :func:`nmbackend.durca.durca_onset` accepts it."""
    n: jnp.ndarray
    x_n: jnp.ndarray
    x_p: jnp.ndarray
    x_e: jnp.ndarray
    x_mu: jnp.ndarray
    mst_n: jnp.ndarray
    mst_p: jnp.ndarray


def cooling_composition(model, params=None) -> CoolingComposition:
    """Composition and Landau masses of ``model`` (a MetaModelEOS, RMFTEOS,
    DDRMFEOS or SkyrmeEOS) at ``params``.  JAX-traceable."""
    c = model.composition(params)
    mn, mp = model.landau_masses(params, c)
    return CoolingComposition(n=c.n, x_n=1.0 - c.x_p, x_p=c.x_p, x_e=c.x_e,
                              x_mu=c.x_mu, mst_n=mn, mst_p=mp)


def pack_d0(cc, n_grid=D0_GRID, dtype=np.float32):
    """(4, len(n_grid)) array of D0_FIELDS, linear in n, NaN outside ``cc.n``."""
    n = np.asarray(cc.n, float)
    return np.stack([np.interp(n_grid, n, np.asarray(getattr(cc, f), float),
                               left=np.nan, right=np.nan)
                     for f in D0_FIELDS]).astype(dtype)


def unpack_d0(arr, n_grid=D0_GRID) -> CoolingComposition:
    """Inverse of :func:`pack_d0` (float64 NumPy arrays)."""
    x_p, x_mu, mn, mp = np.asarray(arr, float)
    return CoolingComposition(n=np.asarray(n_grid, float), x_n=1.0 - x_p,
                              x_p=x_p, x_e=x_p - x_mu, x_mu=x_mu,
                              mst_n=mn, mst_p=mp)
