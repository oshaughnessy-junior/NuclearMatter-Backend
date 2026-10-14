"""EOS base classes and the geometric-unit EOS table consumed by the TOV solver.

An ``EOSTable`` holds JAX arrays describing a cold, barotropic EOS on a baryon
density grid, already converted to geometric units (km^-2).  Everything is
``jax.numpy`` so gradients propagate from EOS *parameters* all the way through
the TOV solve (e.g. dLambda/dL_sym).

Pseudo-enthalpy convention (Lindblom 1992):  h(p) = integral_0^p dp'/(eps+p').

Motivated by ``jester`` (MIT) and RIFT ``EOSManager``; see ATTRIBUTION.md.
"""
from __future__ import annotations

import functools
import math
import warnings
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import MEVFM3_TO_GEOM


class EOSTable(NamedTuple):
    """Barotropic EOS sampled on a baryon-density grid, in geometric units.

    Attributes
    ----------
    n        : baryon number density, fm^-3 (increasing)
    eps      : total energy density, geometric km^-2
    p        : pressure, geometric km^-2
    h        : pseudo-enthalpy (dimensionless), increasing with n
    cs2      : sound speed squared dp/deps at the nodes, by finite differences
               (dimensionless, units of c^2; clipped to [1e-8, 1]).  A
               diagnostic/export field: the TOV tidal term uses the analytic
               deps/dh of the Hermite cell instead (:meth:`at_h_tidal`).
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
        hold by construction.  cs2 is interpolated linearly in h from the node
        values (free to be NON-MONOTONE -- never monotonized); it is a diagnostic,
        not the derivative of this eps(h).  See :func:`hermite_eos_at_h`."""
        return hermite_eos_at_h(self.h, self.p, self.eps, self.cs2, h_query)

    def at_h_tidal(self, h_query):
        """Return (eps, p, deps/dh) at ``h_query``: eps and p as :meth:`at_h`,
        and the analytic deps/dh of the same Hermite cell, which equals the tidal
        term (eps + p)/cs^2.  Used by the TOV solver.  See
        :func:`hermite_eos_tidal_at_h`."""
        return hermite_eos_tidal_at_h(self.h, self.p, self.eps, h_query)

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


def _hermite_cell(h_nodes, p_nodes, eps_nodes, h_query, xp, deps=False):
    """Cubic Hermite of p(h) with node slopes eps + p: (eps, p[, deps/dh]) at
    ``h_query``, eps = dp/dh - p unfloored.  With ``deps``, also the cell's own
    deps/dh = d^2p/dh^2 - dp/dh (see :func:`hermite_eos_tidal_at_h`)."""
    h_nodes = xp.asarray(h_nodes)
    N = h_nodes.shape[0]
    i = xp.clip(xp.searchsorted(h_nodes, h_query), 1, N - 1)
    h0, h1 = h_nodes[i - 1], h_nodes[i]
    dh = h1 - h0
    t = (h_query - h0) / dh
    return _hermite_cell_t(i, t, dh, p_nodes, eps_nodes, deps)


def _hermite_cell_t(i, t, dh, p_nodes, eps_nodes, deps=False):
    """:func:`_hermite_cell` in cell ``i`` (nodes i-1, i) of h-width ``dh`` at the
    cell-local coordinate ``t`` in [0, 1].  The TOV solver calls it with t taken
    on a gradient-stopped node grid (``tov.solver._eos_at_u``)."""
    p0, p1 = p_nodes[i - 1], p_nodes[i]
    e0, e1 = eps_nodes[i - 1], eps_nodes[i]
    s0, s1 = e0 + p0, e1 + p1                      # exact slopes dp/dh = eps + p
    t2, t3 = t * t, t * t * t

    H00, H10 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t
    H01, H11 = -2*t3 + 3*t2, t3 - t2
    p = H00*p0 + H10*dh*s0 + H01*p1 + H11*dh*s1

    dH00, dH10 = 6*t2 - 6*t, 3*t2 - 4*t + 1
    dH11 = 3*t2 - 2*t
    # Combine the opposite pressure terms before division to reduce cancellation.
    dpdh = dH00*(p0-p1) / dh + dH10*s0 + dH11*s1
    eps = dpdh - p                                  # derived: dp/dh = eps + p
    if not deps:
        return eps, p
    # d^2p/dh^2 through the end-slope excesses over the secant S, so the
    # O(s/dh) parts cancel analytically: dh p'' = (6t-4)(s0-S) + (6t-2)(s1-S).
    S = (p1 - p0) / dh
    d2pdh2 = ((6*t - 4)*(s0 - S) + (6*t - 2)*(s1 - S)) / dh
    return eps, p, d2pdh2 - dpdh


def _floor_eps_p(eps, p, xp):
    # Floor with where, not maximum/clip: under jaxlib <= 0.9 the batched JVP of
    # these two maxima inside the TOV loop is miscompiled, so jax.jacfwd (not
    # jax.jvp) of M, R, Lambda came out wrong by 1-20%.  Pinned by
    # tests/test_ad_gradients.py::test_jacfwd_matches_jvp_with_tidal.
    return (xp.where(eps > 1e-300, eps, 1e-300),
            xp.where(p > 1e-300, p, 1e-300))


def hermite_eos_at_h(h_nodes, p_nodes, eps_nodes, cs2_nodes, h_query, xp=jnp):
    """Thermodynamically consistent EOS evaluation at pseudo-enthalpy ``h_query``.

    The *structure* sector uses a cubic-Hermite interpolation of p(h) whose node
    slopes are set to the EXACT thermodynamic value dp/dh = eps + p.  Hence, by
    construction:

      * dp/dh = eps + p exactly at the nodes and smoothly (C^1) between them, so the
        enthalpy-formulation TOV equations are evaluated consistently;
      * eps(h) = dp/dh - p is *derived*, not independently interpolated.

    The returned cs^2 is interpolated linearly in h from the node values
    ``cs2_nodes`` (clipped to [1e-10, 1]); it is left free to be NON-MONOTONE and
    is never monotonized.  It is NOT the derivative of the eps(h) above, and since
    2026-10-14 the TOV tidal term no longer uses it: across a steep cell its
    integral misses part of the cell's Delta eps.  The tidal term takes deps/dh
    from the same Hermite cell instead (:func:`hermite_eos_tidal_at_h`); this
    cs^2 remains for diagnostics (e.g. the central cs^2 of ``tov.central``).

    This uses h (equivalently the baryon chemical potential mu = mu0 e^h) as the
    parameter, which is the natural variable for realistic tabulated EOS.  Fully
    JAX-differentiable; the integer interval index is piecewise-constant (standard
    for interpolation) while gradients flow through the gathered node values.
    ``xp=numpy`` evaluates in NumPy (no JAX compile for new array shapes).
    """
    eps, p = _hermite_cell(h_nodes, p_nodes, eps_nodes, h_query, xp)
    cs2 = xp.clip(xp.interp(h_query, xp.asarray(h_nodes), cs2_nodes), 1e-10, 1.0)
    eps, p = _floor_eps_p(eps, p, xp)
    return eps, p, cs2


def hermite_eos_tidal_at_h(h_nodes, p_nodes, eps_nodes, h_query, xp=jnp):
    """(eps, p, deps/dh) at ``h_query``; eps and p are those of
    :func:`hermite_eos_at_h`, bit for bit.

    deps/dh = d^2p/dh^2 - dp/dh is the analytic derivative of the cell's own
    eps(h) = dp/dh - p.  The tidal equation's term (eps + p)/cs^2 =
    (eps + p) deps/dp equals deps/dh exactly, because dp/dh = eps + p holds in
    the interpolant.  So across every cell the tidal source integrates to the
    Delta eps that the structure equations see, however steep the cell
    (docs/devlog/2026-10-14-tidal-cs2.md).  The node cs^2 interpolated linearly
    in h does not: across a stiff joint it leaves a "phantom" energy jump in y
    and biases Lambda low.

    deps/dh is quadratic in h within a cell and jumps at the nodes (the Hermite
    is C^1); the adaptive TOV solve stops at the nodes (``jump_ts``).  It is
    clipped to [0, 1e10 (eps + p)], i.e. 1/cs^2 in [0, 1e10] (the old cs^2 floor
    was 1e-10).  The clip never acts on the reference tables of the devlog; it
    guards degenerate cells, e.g. a pressure plateau ~1e-16 wide in h (a density
    jump at fixed p, as in some bridge-prior draws), where the cubic's deps/dh
    is +-1e18 and roundoff noise.  Such a jump needs the junction condition; the
    clip drops its Delta eps from the tidal sector, as the old term did."""
    eps, p, deps_dh = _hermite_cell(h_nodes, p_nodes, eps_nodes, h_query, xp,
                                    deps=True)
    eps, p = _floor_eps_p(eps, p, xp)
    return eps, p, _clip_tidal(deps_dh, eps, p, xp)


def _clip_tidal(deps_dh, eps, p, xp):
    """deps/dh clipped to [0, 1e10 (eps + p)] (with where; see _floor_eps_p)."""
    hi = 1e10 * (eps + p)
    deps_dh = xp.where(deps_dh < hi, deps_dh, hi)
    return xp.where(deps_dh > 0.0, deps_dh, 0.0)


def _h_steps_loglinear(eps, p, xp, n_gl=12):
    """Per-cell pseudo-enthalpy steps: int dp/(eps + p) along the log-linear
    (piecewise-polytrope) eps(p) between the two nodes, by Gauss-Legendre.
    Cells with a non-positive eps or p fall back to the trapezoid."""
    x, w = np.polynomial.legendre.leggauss(n_gl)
    s, w = 0.5 * (x + 1.0), 0.5 * w
    ok = (p[1:] > 0) & (p[:-1] > 0) & (eps[1:] > 0) & (eps[:-1] > 0)
    one = xp.ones_like(p[1:])
    lp = xp.log(xp.where(ok, p[1:], one) / xp.where(ok, p[:-1], one))
    le = xp.log(xp.where(ok, eps[1:], one) / xp.where(ok, eps[:-1], one))
    p0 = xp.where(ok, p[:-1], one)[:, None]
    e0 = xp.where(ok, eps[:-1], one)[:, None]
    ps = p0 * xp.exp(s[None, :] * lp[:, None])
    es = e0 * xp.exp(s[None, :] * le[:, None])
    gl = (w[None, :] * ps / (es + ps)).sum(axis=1) * lp       # dp = p lp ds
    integrand = 1.0 / (eps + p)
    trap = 0.5 * (integrand[1:] + integrand[:-1]) * xp.diff(p)
    return xp.where(ok, gl, trap)


def build_h_and_cs2(n, eps_mevfm3, p_mevfm3, h_base=None, xp=jnp, *,
                    h_rule="loglinear"):
    """Given (n, eps, p) build the geometric ``EOSTable`` with h and cs2.

    Parameters in MeV/fm^3 (eps, p) and fm^-3 (n).  Pseudo-enthalpy steps are
    dh = int dp/(eps+p) per cell in geometric units: with ``h_rule="loglinear"``
    (default since 2026-10-14) along the log-linear eps(p) between the nodes
    (12-point Gauss-Legendre); with ``"trapezoid"`` by the trapezoid rule (the
    rule before 2026-10-14).  The Hermite p(h) of :func:`hermite_eos_at_h` takes
    its node slopes from dp/dh = eps + p and its cell widths from these steps;
    the trapezoid width is inconsistent with those slopes by O((Delta ln eps)^2)
    per cell, which made the Hermite eps(h) overshoot between coarse crust nodes
    (R1.4 off by +4 m on the MetaModel and GM1 tables, +41 m on a 250-node SLy
    table, at any core resolution; docs/devlog/2026-10-14-tidal-cs2.md).  cs2 by
    finite difference dp/deps on the grid.  All JAX-differentiable.

    ``h_base`` : if given, the pseudo-enthalpy at the lowest grid point is set to
    this value instead of the small self-consistent offset.  Used to continue the
    enthalpy of an inner (e.g. quark) segment above a phase-transition enthalpy.

    ``xp=numpy`` returns NumPy arrays, for inputs whose length varies from call
    to call (every new length would compile each JAX op); see
    :func:`refine_table`.
    """
    # force jnp so every EOSTable is uniform (numpy inputs -- hybrid combined table,
    # ingested CompOSE/CMF tables -- otherwise break the jit-traced Hermite at_h)
    n = xp.asarray(n)
    eps_mevfm3 = xp.asarray(eps_mevfm3)
    p_mevfm3 = xp.asarray(p_mevfm3)
    eps = eps_mevfm3 * MEVFM3_TO_GEOM            # km^-2
    p = p_mevfm3 * MEVFM3_TO_GEOM                # km^-2

    # sound speed squared dp/deps via central-ish differences (log grid safe)
    cs2 = xp.gradient(p) / xp.gradient(eps)
    cs2 = xp.clip(cs2, 1e-8, 1.0)

    # pseudo-enthalpy: dh = dp/(eps+p).  Integrate from the lowest grid point.
    if h_rule == "loglinear":
        seg = _h_steps_loglinear(eps, p, xp)
    elif h_rule == "trapezoid":
        integrand = 1.0 / (eps + p)
        seg = 0.5 * (integrand[1:] + integrand[:-1]) * xp.diff(p)
    else:
        raise ValueError(f"unknown h_rule {h_rule!r}")
    h_rel = xp.concatenate([xp.zeros((1,)), xp.cumsum(seg)])
    h0 = (p[0] / (eps[0] + p[0])) if h_base is None else h_base
    h = h_rel + h0

    return EOSTable(
        n=n, eps=eps, p=p, h=h, cs2=cs2,
        eps_mevfm3=eps_mevfm3, p_mevfm3=p_mevfm3,
    )


def refine_table(table, n_out):
    """``table`` with nodes added up to ``n_out`` in total, so that tables whose
    node count varies from draw to draw have one length (and the TOV family
    compiles once, not once per length).  All original nodes are kept; each
    added node splits the interval of largest remaining h-width and takes
    (eps, p) from the table's own Hermite p(h) and cs2 from its linear
    interpolant, so :meth:`EOSTable.at_h` is unchanged up to roundoff.  n is
    log-linear in h.  Works in NumPy (pass a table from
    ``build_h_and_cs2(..., xp=numpy)`` to avoid JAX ops on the variable
    length) and returns JAX arrays; a table with ``n_out`` or more nodes only
    changes array type (with a RuntimeWarning if more)."""
    t = EOSTable(*(np.asarray(a) for a in table))
    m = n_out - t.h.size
    if m < 0:
        warnings.warn(f"refine_table: {t.h.size} nodes > n_out={n_out}; the "
                      "table keeps its own length (one compile per length)",
                      RuntimeWarning, stacklevel=2)
    if m > 0:
        dh = np.diff(t.h)
        c = np.zeros(dh.size, int)               # added nodes per interval
        for _ in range(m):
            c[np.argmax(dh / (c + 1))] += 1
        h_new = np.concatenate([t.h[i] + dh[i] * np.arange(1, c[i] + 1) / (c[i] + 1)
                                for i in np.flatnonzero(c)])
        order = np.argsort(np.concatenate([t.h, h_new]), kind="stable")
        eps_a, p_a, cs2_a = hermite_eos_at_h(t.h, t.p, t.eps, t.cs2, h_new, xp=np)
        n_a = np.exp(np.interp(h_new, t.h, np.log(t.n)))
        cat = lambda a, b: np.concatenate([a, b])[order]
        eps, p = cat(t.eps, eps_a), cat(t.p, p_a)
        t = EOSTable(n=cat(t.n, n_a), eps=eps, p=p, h=cat(t.h, h_new),
                     cs2=cat(t.cs2, cs2_a), eps_mevfm3=eps / MEVFM3_TO_GEOM,
                     p_mevfm3=p / MEVFM3_TO_GEOM)
    return EOSTable(*(jnp.asarray(a) for a in t))


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

    def _config(self):
        """Hashable snapshot of the instance attributes (arrays as tuples); the
        key of the table cached by :func:`compiled_table`."""
        def key(v):
            if hasattr(v, "shape"):
                return tuple(np.asarray(v).ravel().tolist())
            try:
                hash(v)
                return v
            except TypeError:
                return (type(v).__name__, repr(v))
        return tuple(sorted((k, key(v)) for k, v in vars(self).items()))


class _ByConfig:
    """Static jit argument standing for a model: equal for models of one type
    and ``_config()``, so all such instances share one compiled table."""
    __slots__ = ("model", "key")

    def __init__(self, model):
        self.model, self.key = model, (type(model), model._config())

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, _ByConfig) and self.key == other.key


@functools.partial(jax.jit, static_argnums=(0, 1))
def _method_jit(handle, name, *args):
    return getattr(handle.model, name)(*args)


def _floats(a):
    """Numeric params as float arrays (one trace for int, float and array
    inputs); str and None entries (labels) are dropped, as no model reads
    them.  Non-dict arguments pass through."""
    if not isinstance(a, dict):
        return a
    return {k: jnp.asarray(v, float) for k, v in a.items()
            if not (v is None or isinstance(v, (str, bytes)))}


def compiled_method(model, name, *args):
    """``getattr(model, name)(*args)`` compiled once per model type,
    ``_config()``, method and argument structure (dict arguments become float
    arrays).  Eager fori_loop bisections recompile on every call (the loop
    body is a new closure each time), and each executable holds ~100 memory
    maps."""
    return _method_jit(_ByConfig(model), name, *(_floats(a) for a in args))


def compiled_table(model, p):
    """``model._table_flags(p)`` through :func:`compiled_method`, then the eager
    rejection of :func:`eos.crust.raise_rejected`."""
    from .crust import raise_rejected
    return raise_rejected(*compiled_method(model, "_table_flags", p))
