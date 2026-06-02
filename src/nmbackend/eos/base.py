"""EOS base classes and the geometric-unit EOS table consumed by the TOV solver.

An ``EOSTable`` holds JAX arrays describing a cold, barotropic EOS on a baryon
density grid, already converted to geometric units (km^-2).  Everything is
``jax.numpy`` so gradients propagate from EOS *parameters* all the way through
the TOV solve (e.g. dLambda/dL_sym).

Pseudo-enthalpy convention (Lindblom 1992):  h(p) = integral_0^p dp'/(eps+p').

Motivated by ``jester`` (MIT) and RIFT ``EOSManager``; see ATTRIBUTION.md.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..constants import MEVFM3_TO_GEOM


class EOSTable(NamedTuple):
    """Barotropic EOS sampled on a baryon-density grid, in geometric units.

    Attributes
    ----------
    n        : baryon number density, fm^-3 (increasing)
    eps      : total energy density, geometric km^-2
    p        : pressure, geometric km^-2
    h        : pseudo-enthalpy (dimensionless), increasing with n
    cs2      : sound speed squared dp/deps (dimensionless, units of c^2)
    eps_mevfm3 : energy density in MeV/fm^3 (handy for I/O / RIFT adapter)
    p_mevfm3   : pressure in MeV/fm^3
    """
    n: jnp.ndarray
    eps: jnp.ndarray
    p: jnp.ndarray
    h: jnp.ndarray
    cs2: jnp.ndarray
    eps_mevfm3: jnp.ndarray
    p_mevfm3: jnp.ndarray

    @property
    def h_max(self) -> jnp.ndarray:
        return self.h[-1]

    # --- differentiable interpolation as a function of pseudo-enthalpy ---
    def at_h(self, h_query):
        """Return (eps, p, cs2) interpolated at pseudo-enthalpy ``h_query``.

        Log-linear in (eps, p) vs h; linear (clamped) in cs2.  ``jnp.interp`` is
        differentiable, so this is safe inside the TOV scan.
        """
        h = self.h
        logeps = jnp.interp(h_query, h, jnp.log(self.eps))
        logp = jnp.interp(h_query, h, jnp.log(self.p))
        cs2 = jnp.interp(h_query, h, self.cs2)
        cs2 = jnp.clip(cs2, 1e-8, 1.0)
        return jnp.exp(logeps), jnp.exp(logp), cs2


def build_h_and_cs2(n, eps_mevfm3, p_mevfm3, h_base=None):
    """Given (n, eps, p) build the geometric ``EOSTable`` with h and cs2.

    Parameters in MeV/fm^3 (eps, p) and fm^-3 (n).  Pseudo-enthalpy is computed
    by cumulative trapezoid integration of dp/(eps+p) in geometric units; cs2 by
    finite difference dp/deps on the grid.  All JAX-differentiable.

    ``h_base`` : if given, the pseudo-enthalpy at the lowest grid point is set to
    this value instead of the small self-consistent offset.  Used to continue the
    enthalpy of an inner (e.g. quark) segment above a phase-transition enthalpy.
    """
    eps = eps_mevfm3 * MEVFM3_TO_GEOM            # km^-2
    p = p_mevfm3 * MEVFM3_TO_GEOM                # km^-2

    # sound speed squared dp/deps via central-ish differences (log grid safe)
    cs2 = jnp.gradient(p) / jnp.gradient(eps)
    cs2 = jnp.clip(cs2, 1e-8, 1.0)

    # pseudo-enthalpy: dh = dp/(eps+p).  Integrate from the lowest grid point.
    integrand = 1.0 / (eps + p)
    dp = jnp.diff(p)
    seg = 0.5 * (integrand[1:] + integrand[:-1]) * dp
    h_rel = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(seg)])
    h0 = (p[0] / (eps[0] + p[0])) if h_base is None else h_base
    h = h_rel + h0

    return EOSTable(
        n=n, eps=eps, p=p, h=h, cs2=cs2,
        eps_mevfm3=eps_mevfm3, p_mevfm3=p_mevfm3,
    )


class EOSModel:
    """Abstract base for parameterized EOS models.

    Subclasses implement :meth:`eos_table` returning an :class:`EOSTable` for a
    given parameter dictionary.  Keeping the parameter -> table map pure and
    JAX-traceable is what makes the whole pipeline differentiable.
    """

    #: human-readable model name
    name: str = "EOSModel"

    def eos_table(self, params: dict) -> EOSTable:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, params: dict) -> EOSTable:
        return self.eos_table(params)
