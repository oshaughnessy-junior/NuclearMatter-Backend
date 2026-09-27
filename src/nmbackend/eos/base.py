"""EOS base classes and the geometric-unit EOS table consumed by the TOV solver.

An ``EOSTable`` holds JAX arrays describing a cold, barotropic EOS on a baryon
density grid, already converted to geometric units (km^-2).  Everything is
``jax.numpy`` so gradients propagate from EOS *parameters* all the way through
the TOV solve (e.g. dLambda/dL_sym).

Pseudo-enthalpy convention (Lindblom 1992):  h(p) = integral_0^p dp'/(eps+p').

Motivated by ``jester`` (MIT) and RIFT ``EOSManager``; see ATTRIBUTION.md.
"""
from __future__ import annotations

import math
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

    @property
    def mu(self) -> jnp.ndarray:
        """Baryon chemical potential mu = (eps+p)/n (geometric km^-1 * ... ); the
        pseudo-enthalpy is h = ln(mu/mu_surface), so mu = mu[0] * exp(h).  Either
        h or mu can be used as the (equivalent) thermodynamic parameter."""
        return (self.eps + self.p) / self.n

    # --- thermodynamically consistent interpolation in pseudo-enthalpy ---
    def at_h(self, h_query):
        """Return (eps, p, cs2) at pseudo-enthalpy ``h_query``, thermodynamically
        consistent in the *structure* sector: p and eps come from a Hermite of p(h)
        with node slopes dp/dh = eps + p, so eps = dp/dh - p and dp/dh = eps + p
        hold by construction.  cs2 is interpolated from accurate node values (it is
        consistent at table resolution and free to be NON-MONOTONE -- never
        monotonized).  See :func:`hermite_eos_at_h`."""
        return hermite_eos_at_h(self.h, self.p, self.eps, self.cs2, h_query)

    def at_h_loglinear(self, h_query):
        """Legacy independent log-linear interpolation (NOT thermodynamically
        consistent: dp/dh != eps+p between nodes; cs2 interpolated+clipped).  Kept
        for regression comparison only."""
        h = self.h
        logeps = jnp.interp(h_query, h, jnp.log(self.eps))
        logp = jnp.interp(h_query, h, jnp.log(self.p))
        cs2 = jnp.clip(jnp.interp(h_query, h, self.cs2), 1e-8, 1.0)
        return jnp.exp(logeps), jnp.exp(logp), cs2


class Composition(NamedTuple):
    """beta-equilibrium particle fractions per baryon on a density grid.

    Charge neutrality: x_p = x_e + x_mu.  ``n`` is the baryon density (fm^-3)
    of the EOS model's own grid (before any surface continuation).
    """
    n: jnp.ndarray
    x_p: jnp.ndarray
    x_e: jnp.ndarray
    x_mu: jnp.ndarray


def hermite_eos_at_h(h_nodes, p_nodes, eps_nodes, cs2_nodes, h_query):
    """Thermodynamically consistent EOS evaluation at pseudo-enthalpy ``h_query``.

    The *structure* sector uses a cubic-Hermite interpolation of p(h) whose node
    slopes are set to the EXACT thermodynamic value dp/dh = eps + p.  Hence, by
    construction:

      * dp/dh = eps + p exactly at the nodes and smoothly (C^1) between them, so the
        enthalpy-formulation TOV equations are evaluated consistently;
      * eps(h) = dp/dh - p is *derived*, not independently interpolated.

    cs^2 is interpolated from the (accurate) node values ``cs2_nodes`` -- it is
    consistent at the table nodes and is left free to be NON-MONOTONE (e.g. dipping
    toward zero at a phase transition); we never monotonize it.  We deliberately do
    NOT take cs^2 from d^2p/dh^2: a second derivative of a C^1 interpolant is far too
    noisy, especially for parameter gradients.

    This uses h (equivalently the baryon chemical potential mu = mu0 e^h) as the
    parameter, which is the natural variable for realistic tabulated EOS.  Fully
    JAX-differentiable; the integer interval index is piecewise-constant (standard
    for interpolation) while gradients flow through the gathered node values.
    """
    h_nodes = jnp.asarray(h_nodes)
    N = h_nodes.shape[0]
    i = jnp.clip(jnp.searchsorted(h_nodes, h_query), 1, N - 1)
    h0, h1 = h_nodes[i - 1], h_nodes[i]
    p0, p1 = p_nodes[i - 1], p_nodes[i]
    e0, e1 = eps_nodes[i - 1], eps_nodes[i]
    s0, s1 = e0 + p0, e1 + p1                      # exact slopes dp/dh = eps + p
    dh = h1 - h0
    t = (h_query - h0) / dh
    t2, t3 = t * t, t * t * t

    H00, H10 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t
    H01, H11 = -2*t3 + 3*t2, t3 - t2
    p = H00*p0 + H10*dh*s0 + H01*p1 + H11*dh*s1

    dH00, dH10 = 6*t2 - 6*t, 3*t2 - 4*t + 1
    dH01, dH11 = -6*t2 + 6*t, 3*t2 - 2*t
    dpdh = (dH00*p0 + dH01*p1) / dh + dH10*s0 + dH11*s1
    eps = dpdh - p                                  # derived: dp/dh = eps + p

    cs2 = jnp.clip(jnp.interp(h_query, h_nodes, cs2_nodes), 1e-10, 1.0)
    # Floor with where, not maximum/clip: under jaxlib <= 0.9 the batched JVP of
    # these two maxima inside the TOV loop is miscompiled, so jax.jacfwd (not
    # jax.jvp) of M, R, Lambda came out wrong by 1-20%.  Pinned by
    # tests/test_ad_gradients.py::test_jacfwd_matches_jvp_with_tidal.
    eps = jnp.where(eps > 1e-300, eps, 1e-300)
    p = jnp.where(p > 1e-300, p, 1e-300)
    return eps, p, cs2


def build_h_and_cs2(n, eps_mevfm3, p_mevfm3, h_base=None):
    """Given (n, eps, p) build the geometric ``EOSTable`` with h and cs2.

    Parameters in MeV/fm^3 (eps, p) and fm^-3 (n).  Pseudo-enthalpy is computed
    by cumulative trapezoid integration of dp/(eps+p) in geometric units; cs2 by
    finite difference dp/deps on the grid.  All JAX-differentiable.

    ``h_base`` : if given, the pseudo-enthalpy at the lowest grid point is set to
    this value instead of the small self-consistent offset.  Used to continue the
    enthalpy of an inner (e.g. quark) segment above a phase-transition enthalpy.
    """
    # force jnp so every EOSTable is uniform (numpy inputs -- hybrid combined table,
    # ingested CompOSE/CMF tables -- otherwise break the jit-traced Hermite at_h)
    n = jnp.asarray(n)
    eps_mevfm3 = jnp.asarray(eps_mevfm3)
    p_mevfm3 = jnp.asarray(p_mevfm3)
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


def extend_to_surface(n, eps_mevfm3, p_mevfm3, n_surface=1e-7, gamma_crust=1.3,
                      n_pts=120):
    """Prepend a thermodynamically consistent low-density tail down to ``n_surface``.

    The TOV radius depends on integrating the star out to (near) zero pressure.  A
    table that stops at, e.g., n_min ~ 3e-4 fm^-3 (just below neutron drip)
    truncates the outer crust/atmosphere and **underestimates R by ~100 m** -- a
    surface *convention* mismatch with reference solvers (RePrimAnd) that integrate
    closer to p -> 0.  Here we continue the EOS below its base point as a single
    polytrope p = p0 (n/n0)^gamma_crust, with the energy density fixed by the cold
    first law deps/dn = (eps+p)/n, whose closed-form solution::

        eps(n) = A n + p/(gamma_crust - 1),   A = (eps0 - p0/(gamma_crust-1))/n0

    is continuous at the base point n0.

    This is a *surface continuation*, NOT a physical crust (no composition / pasta);
    ``gamma_crust`` and ``n_surface`` are the documented systematic knobs.  All
    JAX-differentiable (the tail tracks the base point, which depends on EOS params).
    """
    n = jnp.asarray(n)
    eps = jnp.asarray(eps_mevfm3)
    p = jnp.asarray(p_mevfm3)
    n0, e0, p0 = n[0], eps[0], p[0]
    A = (e0 - p0 / (gamma_crust - 1.0)) / n0
    n_lo = jnp.exp(jnp.linspace(math.log(n_surface), jnp.log(n0), n_pts))[:-1]
    p_lo = p0 * (n_lo / n0) ** gamma_crust
    e_lo = A * n_lo + p_lo / (gamma_crust - 1.0)
    return (jnp.concatenate([n_lo, n]),
            jnp.concatenate([e_lo, eps]),
            jnp.concatenate([p_lo, p]))


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
