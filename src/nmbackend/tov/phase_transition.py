"""TOV solver for EOS with first-order phase transitions (density discontinuities).

A first-order transition is a density jump at constant pressure.  In the
pseudo-enthalpy variable the transition sits at a single enthalpy ``h_t`` (because
dh = dp/(eps+p) and dp = 0 across the jump).  The metric perturbation H is
continuous there, but the tidal variable y is **discontinuous** -- the
(eps+p)/cs^2 term in the y-equation becomes a delta function as cs^2 -> 0.
Integrating across it gives the junction condition (Damour & Nagar 2009;
Postnikov, Prakash & Lattimer 2010; Takatsy & Kovacs 2020):

    y_out = y_in - 4 pi r_d^3 (eps_out - eps_in) / (m_d + 4 pi r_d^3 p_t)

with r_d, m_d the radius/mass at the transition, ``in`` the inner (higher-density)
side and ``out`` the outer (lower-density) side.  Since eps_out < eps_in this
raises y across the transition going outward.

We integrate each smooth phase as a separate adaptive (diffrax) segment and apply
the junction between them, so the genuine matter sound speed is used inside each
phase and the discontinuity is handled analytically (no cs^2 clipping/smearing).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .solver import (_field_tidal, _k2_from_y_C, _central_start, TOVPoint,
                     _HAVE_DIFFRAX, solve_structure)
from ..constants import GMSUN_KM

if _HAVE_DIFFRAX:
    import diffrax

_FOURPI = 4.0 * math.pi


def _diffrax_segment(table, t0, t1, state0, rtol, atol, adjoint=None,
                     max_steps=100000):
    """Integrate (r, m, y) in enthalpy from t0 down to t1 with a given start."""
    t0, t1 = jnp.asarray(t0), jnp.asarray(t1)   # a Python float here is a jit key
    term = diffrax.ODETerm(_field_tidal)
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    if adjoint is None:
        adjoint = diffrax.ForwardMode()
    sol = diffrax.diffeqsolve(
        term, diffrax.Dopri5(), t0=t0, t1=t1, dt0=-(t0 - t1) * 1e-3,
        y0=state0, args=table, stepsize_controller=ctrl, max_steps=max_steps,
        adjoint=adjoint,
    )
    return sol.ys[0]


def solve_hybrid(hybrid, h_c, rtol: float = 1e-7, atol: float = 1e-9) -> TOVPoint:
    """TOV solve for a :class:`~nmbackend.eos.hybrid.HybridEOS` (one transition).

    Applies the tidal junction condition at the density discontinuity.
    """
    if not _HAVE_DIFFRAX:  # pragma: no cover
        raise ImportError("solve_hybrid requires diffrax")
    inner, outer = hybrid.segments
    tr = hybrid.transition
    h_c = float(h_c)

    # low-mass star with no quark core: purely hadronic
    if h_c <= tr.h_t:
        return solve_structure(outer, h_c, backend="diffrax", rtol=rtol, atol=atol)

    # 1) inner (quark) core: central series start -> integrate down to h_t
    _, _, dh0, state0 = _central_start(inner, h_c)
    state_t = _diffrax_segment(inner, h_c - dh0, tr.h_t, state0, rtol, atol)
    r_d, m_d, y_in = state_t[0], state_t[1], state_t[2]

    # 2) tidal junction at the discontinuity
    y_out = y_in - _FOURPI * r_d**3 * (tr.eps_out - tr.eps_in) / (
        m_d + _FOURPI * r_d**3 * tr.p_t)

    # 3) outer (hadronic) envelope: h_t -> surface, starting from junction state
    state_surf = _diffrax_segment(
        outer, tr.h_t, outer.h[0], jnp.stack([r_d, m_d, y_out]), rtol, atol)
    R, M, y = state_surf[0], state_surf[1], state_surf[2]
    C = M / R
    k2 = _k2_from_y_C(y, C)
    Lam = (2.0 / 3.0) * k2 / C**5
    return TOVPoint(h_c=jnp.asarray(h_c), R_km=R, M_km=M, M_msun=M / GMSUN_KM,
                    C=C, y=y, k2=k2, Lambda=Lam)


def hybrid_sequence(hybrid, h_c_array, rtol=1e-7, atol=1e-9):
    """List of TOVPoint over central enthalpies (handles the transition)."""
    return [solve_hybrid(hybrid, float(hc), rtol, atol) for hc in h_c_array]
