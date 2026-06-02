"""Differentiable TOV solver in pseudo-enthalpy, with in-loop tidal Love number.

We integrate the TOV equations using the pseudo-enthalpy ``h`` as the independent
variable (Lindblom 1992), which removes the surface coordinate singularity and
makes the central boundary condition a simple series start.  Alongside (r, m) we
carry the tidal variable y (Postnikov, Prakash & Lattimer 2010) to obtain the
l=2 Love number k2 and dimensionless tidal deformability Lambda.

Two backends sit behind one seam, ``solve_structure``:

* ``backend="diffrax"`` (default): **adaptive** Dopri5 + PID controller.  This is
  the accurate, recommended path -- it automatically resolves the stiff regions
  near the centre (1/r terms) *and* near the surface (which the tidal sector needs)
  and is validated against an independent radius-domain solver to <0.1% in M, R,
  and Lambda (see ``tests/test_validation_reference.py``).  Forward-mode AD flows
  through it via ``diffrax.ForwardMode`` -- this is the supported sensitivity path
  and reproduces the perturbation ("delta-TOV") equations.

* ``backend="scan"``: a lightweight fixed-step RK4 ``jax.lax.scan``.  Fully
  jit/vmap/grad-able (incl. reverse-mode for structure-only, ``tidal=False``), but
  a *fixed* grid is only accurate with enough, well-placed steps -- prefer the
  diffrax backend unless you specifically need the dependency-free scan.

WARNING (see docs/validation-plan.md): neither backend yet handles strong
first-order phase transitions (cs^2 -> 0 plateaus / density discontinuities) with
their tidal junction conditions.  Validate any phase-transition EOS against
RePrimAnd before trusting it.

Geometric units throughout: r, m in km; eps, p in km^-2.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

try:
    import diffrax
    _HAVE_DIFFRAX = True
except Exception:  # pragma: no cover
    _HAVE_DIFFRAX = False

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
        # structure-only: keep y constant so its singular near-centre terms are
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


def _central_start(table, h_c, frac=1e-8):
    h_min = table.h[0]
    h_c = jnp.asarray(h_c)
    dh0 = (h_c - h_min) * frac
    eps_c, p_c, _ = table.at_h(h_c)
    r0 = jnp.sqrt(3.0 * dh0 / (2.0 * math.pi * (eps_c + 3.0 * p_c)))
    m0 = (4.0 / 3.0) * math.pi * eps_c * r0**3
    return h_min, h_c, dh0, jnp.stack([r0, m0, jnp.asarray(2.0)])


def _finish(R, M, y, h_c, tidal):
    C = M / R
    if tidal:
        k2 = _k2_from_y_C(y, C)
        Lam = (2.0 / 3.0) * k2 / C**5
    else:
        k2 = jnp.nan * R
        Lam = jnp.nan * R
    return TOVPoint(h_c=h_c, R_km=R, M_km=M, M_msun=M / GMSUN_KM,
                    C=C, y=y, k2=k2, Lambda=Lam)


# ----- fixed-step RK4 scan backend -----
def _solve_scan(table, h_c, n_steps, tidal):
    h_min, h_c, dh0, state0 = _central_start(table, h_c)
    hs = jnp.linspace(h_c - dh0, h_min, n_steps + 1)
    h_lefts, dhs = hs[:-1], jnp.diff(hs)

    def step(state, hd):
        h0, dh = hd
        k1 = _derivs(h0, state, table, tidal)
        k2 = _derivs(h0 + 0.5 * dh, state + 0.5 * dh * k1, table, tidal)
        k3 = _derivs(h0 + 0.5 * dh, state + 0.5 * dh * k2, table, tidal)
        k4 = _derivs(h0 + dh, state + dh * k3, table, tidal)
        return state + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    state, _ = jax.lax.scan(step, state0, (h_lefts, dhs))
    return _finish(state[0], state[1], state[2], h_c, tidal)


# ----- adaptive diffrax backend -----
def _solve_diffrax(table, h_c, rtol, atol, tidal, adjoint=None, max_steps=100000):
    if not _HAVE_DIFFRAX:  # pragma: no cover
        raise ImportError("backend='diffrax' requires the 'diffrax' package; "
                          "pip install diffrax  (or use backend='scan').")
    h_min, h_c, dh0, state0 = _central_start(table, h_c)

    def field(h, s, args):
        return _derivs(h, s, args, tidal)

    term = diffrax.ODETerm(field)
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    if adjoint is None:
        adjoint = diffrax.ForwardMode()   # supports values, vmap, forward-mode AD
    sol = diffrax.diffeqsolve(
        term, diffrax.Dopri5(), t0=h_c - dh0, t1=h_min,
        dt0=-(h_c - h_min) * 1e-4, y0=state0, args=table,
        stepsize_controller=ctrl, max_steps=max_steps, adjoint=adjoint,
    )
    ys = sol.ys[0]
    return _finish(ys[0], ys[1], ys[2], h_c, tidal)


def solve_structure(table: EOSTable, h_c, *, backend: str = "diffrax",
                    tidal: bool = True, rtol: float = 1e-7, atol: float = 1e-9,
                    n_steps: int = 2000, adjoint=None):
    """Integrate one star at central pseudo-enthalpy ``h_c``.  Returns TOVPoint.

    Backend seam: callers depend only on this signature and ``TOVPoint``.

    Parameters
    ----------
    backend : "diffrax" (adaptive, accurate, forward-mode AD; default) or
              "scan" (fixed-step RK4; lightweight, reverse-mode-capable).
    tidal   : integrate the tidal sector (k2, Lambda).  ``tidal=False`` drops y so
              reverse-mode AD of (R, M) is clean (scan backend).
    rtol, atol : diffrax tolerances.   n_steps : scan-backend step count.
    adjoint : optional diffrax adjoint override (default ForwardMode()).
    """
    if backend == "diffrax":
        return _solve_diffrax(table, h_c, rtol, atol, tidal, adjoint=adjoint)
    elif backend == "scan":
        return _solve_scan(table, h_c, n_steps, tidal)
    raise ValueError(f"unknown backend {backend!r} (use 'diffrax' or 'scan')")
