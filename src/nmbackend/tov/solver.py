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
  and reproduces the perturbation ("delta-TOV") equations.  Two mechanisms make
  the tangents accurate (docs/devlog/2026-10-12-tov-jump-ts.md):

  - steps stop at the EOS table nodes (diffrax ``jump_ts``), where the
    right-hand side has slope kinks.  This fixes the EOS-*parameter* tangents,
    which now converge with rtol at the level of the values (~3e-5 at the
    default rtol, ~1e-7 at 1e-9; up to 10% before).
  - the start of the integration is frozen against differentiation.  This fixes
    the *h_c* tangent (3.6e-2 -> 1.2e-4 at the default rtol).  It converges like
    rtol but sits ~1e3x above the values (~1e-4 at the default rtol, ~5e-7 at
    1e-9), from the near-centre resolution; ``centre_refine=True`` buys ~10x.
    The residual is a near-common factor on M, R, Lambda, so it largely cancels
    in fixed-mass derivatives.

  FIRST derivatives only w.r.t. h_c: d^2/dh_c^2 is wrong by orders of
  magnitude (mixed d^2/dh_c dtheta is fine).

* ``backend="scan"``: a lightweight fixed-step RK4 ``jax.lax.scan``.  Fully
  jit/vmap/grad-able (incl. reverse-mode for structure-only, ``tidal=False``), but
  a *fixed* grid is only accurate with enough, well-placed steps -- prefer the
  diffrax backend unless you specifically need the dependency-free scan.  Its
  uniform grid ignores the table nodes and moves with h_c; its tangent is the
  exact derivative of that discrete map, so it agrees with FD of the same scan
  but carries the grid's (low-order, node-straddling) discretization error.

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


class TOVProfile(NamedTuple):
    """Structure-only profile, centre-series start to surface; r and m in km.

    Enthalpy decreases outward. The first point is at the small nonzero
    central-series radius. Redshift
    uses the same finite-pressure surface convention as solve_structure.
    """
    h: jnp.ndarray
    r_km: jnp.ndarray
    m_km: jnp.ndarray
    ephi: jnp.ndarray



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
# Module-level vector fields: diffeqsolve is jitted with the term as a static key, so a
# fresh closure per call would recompile every solve and leak the executables.
def _field_tidal(h, s, args):
    return _derivs(h, s, args, True)


def _field_structure(h, s, args):
    return _derivs(h, s, args, False)


def _node_jumps(table):
    """Interior EOS-table nodes as diffrax ``jump_ts`` (gradient-stopped).

    The TOV right-hand side is only C^0 in h at every table node: eps comes from
    a C^1 Hermite of p(h) (so eps' jumps) and cs2 is linearly interpolated.  An
    adaptive step that straddles a node rejects repeatedly there, and ForwardMode
    then differentiates the mesh's position relative to the kink -- a tangent
    bias that shrinks only slowly and erratically with rtol (docs/devlog/
    2026-10-12-tov-jump-ts.md).  Stepping to just before and restarting just after
    each node removes both.  ``table.h[0]`` is the surface (the integration end
    point) and is left out; nodes outside the integration interval are inert, so
    the array keeps the table's static length under jit/vmap.  The positions are
    gradient-stopped: the mesh is fixed, and the node motion enters the tangent
    through the vector field on each side of the node, where it belongs."""
    return jax.lax.stop_gradient(table.h[1:])


def _adaptive_tov(table, t0, t1, state0, rtol, atol, tidal, dt0,
                  adjoint=None, max_steps=100000, saveat=None, jump_nodes=True):
    """Adaptive Dopri5 integration of the TOV(+tidal) system from ``t0`` down to
    ``t1`` in pseudo-enthalpy, with steps cut at the EOS table nodes.

    Two mechanisms, each fixing one column of the tangent (docs/devlog/
    2026-10-12-tov-jump-ts.md):

    * ``jump_ts`` at the table nodes fixes the *parameter* tangents: the mesh
      no longer straddles kinks whose position moves with the EOS parameters.
    * The frozen start fixes the *h_c* tangent and is REQUIRED whenever
      ``jump_ts`` is on.  With a live t0/dt0 the mesh moves with h_c while the
      (gradient-stopped) node clips do not, and the h_c tangent error grows as
      rtol tightens (0.49 / 4.8e3 / 1.7e7 at rtol 1e-7 / 1e-8 / 1e-9).

    The start time and the initial step are frozen against differentiation, so
    no mesh point before the last carries a tangent.  The start's dependence on
    the inputs moves into the starting state through the flow expansion
    ``y(t0s) = y(t0) + f d + f' d^2 / 2`` with ``d = t0s - t0 == 0`` and
    ``f' = df/dt`` along the flow: values are unchanged, first tangents are
    exact (``dy0 - f dt0``), and the ``f' dt0^2`` term of the second tangent is
    kept.  ``t1`` stays live: the last step ends there and spans no node.

    Derivatives w.r.t. the start (h_c) are reliable to FIRST order only.  Second
    derivatives in h_c are wrong by orders of magnitude (the frozen mesh
    under-resolves the (h_c - h)^-3/2 second tangent near the centre);
    mixed d^2/dh_c dtheta and all parameter derivatives are fine.
    ``tests/test_tov_jump_ts.py::test_second_hc_derivative_is_not_supported``
    pins this."""
    sg = jax.lax.stop_gradient
    t0, t1 = jnp.asarray(t0), jnp.asarray(t1)   # a Python float here is a jit key
    t0s = sg(t0)                                # REQUIRED with jump_ts; see above
    f0 = _derivs(t0, state0, table, tidal)
    _, fdot = jax.jvp(lambda t, y: _derivs(t, y, table, tidal),
                      (t0, state0), (jnp.ones_like(t0), f0))
    d = t0s - t0
    state0 = state0 + f0 * d + 0.5 * fdot * d * d
    term = diffrax.ODETerm(_field_tidal if tidal else _field_structure)
    ctrl = diffrax.PIDController(
        rtol=rtol, atol=atol,
        jump_ts=_node_jumps(table) if jump_nodes else None)
    if adjoint is None:
        adjoint = diffrax.ForwardMode()   # supports values, vmap, forward-mode AD
    return diffrax.diffeqsolve(
        term, diffrax.Dopri5(), t0=t0s, t1=t1, dt0=sg(dt0),
        y0=state0, args=table, stepsize_controller=ctrl, max_steps=max_steps,
        adjoint=adjoint,
        saveat=diffrax.SaveAt(t1=True) if saveat is None else saveat,
    )


#: ``centre_refine``: the inner fraction of (h_c - h_min) integrated at a
#: tolerance tighter by ``_CENTRE_TOL`` (floored at min(rtol, ``_CENTRE_RTOL_MIN``))
_CENTRE_FRAC = 0.05
_CENTRE_TOL = 1e-3
_CENTRE_RTOL_MIN = 1e-13


def _integrate_diffrax(table, h_c, rtol, atol, tidal, adjoint=None,
                       max_steps=100000, saveat=None, *, jump_nodes=True,
                       centre_refine=False):
    if not _HAVE_DIFFRAX:  # pragma: no cover
        raise ImportError("backend='diffrax' requires the 'diffrax' package; "
                          "pip install diffrax  (or use backend='scan').")
    h_min, h_c, dh0, state0 = _central_start(table, h_c)
    t0, dt0 = h_c - dh0, -(h_c - h_min) * 1e-4
    if not centre_refine:
        return _adaptive_tov(table, t0, h_min, state0, rtol, atol, tidal, dt0,
                             adjoint, max_steps, saveat, jump_nodes)
    if saveat is not None:
        raise ValueError("centre_refine supports SaveAt(t1=True) only")
    # Inner shell to a frozen split point at a tighter tolerance, then the rest.
    # The split is an arbitrary interior point, so freezing it needs no
    # correction; the state carries the tangents across.
    h_split = jax.lax.stop_gradient(t0 - _CENTRE_FRAC * (h_c - h_min))
    # inner rtol = max(rtol * 1e-3, min(rtol, 1e-13)): never looser than rtol
    scale = jnp.maximum(_CENTRE_TOL, jnp.minimum(1.0, _CENTRE_RTOL_MIN / rtol))
    inner = _adaptive_tov(table, t0, h_split, state0, rtol * scale,
                          atol * scale, tidal, dt0, adjoint, max_steps, None,
                          jump_nodes)
    return _adaptive_tov(table, h_split, h_min, inner.ys[0], rtol, atol, tidal,
                         10.0 * dt0, adjoint, max_steps, None, jump_nodes)


def _solve_diffrax(table, h_c, rtol, atol, tidal, adjoint=None, max_steps=100000,
                   *, jump_nodes=True, centre_refine=False):
    sol = _integrate_diffrax(table, h_c, rtol, atol, tidal, adjoint, max_steps,
                             jump_nodes=jump_nodes, centre_refine=centre_refine)
    ys = sol.ys[0]
    return _finish(ys[0], ys[1], ys[2], h_c, tidal)


def solve_profile(table: EOSTable, h_c, *, n_points: int = 256,
                  rtol: float = 1e-7, atol: float = 1e-9, adjoint=None):
    """Save an adaptive spherical TOV profile for thermal volume integrals.

    Quadratic spacing in enthalpy drop resolves the central r ~ sqrt(h_c-h)
    behavior. Saving points does not change the adaptive integration mesh.
    Default ForwardMode supports JVP/jacfwd through central enthalpy
    and smooth table inputs, with n_points fixed.
    Smooth EOS only; phase-transition junctions are not included.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if not _HAVE_DIFFRAX:  # pragma: no cover
        raise ImportError("solve_profile requires diffrax")
    h_min, h_c, dh0, _ = _central_start(table, h_c)
    h_start = h_c - dh0
    h = h_start + (h_min - h_start) * jnp.linspace(0., 1., n_points)**2
    h = h.at[0].set(h_start).at[-1].set(h_min)
    sol = _integrate_diffrax(table, h_c, rtol, atol, False, adjoint,
                            saveat=diffrax.SaveAt(ts=h))
    r, m = sol.ys[:, 0], sol.ys[:, 1]
    ephi_surface = jnp.sqrt(1. - 2. * m[-1] / r[-1])
    ephi = ephi_surface * jnp.exp(h_min - h)
    return TOVProfile(h, r, m, ephi)


def solve_structure(table: EOSTable, h_c, *, backend: str = "diffrax",
                    tidal: bool = True, rtol: float = 1e-7, atol: float = 1e-9,
                    n_steps: int = 2000, adjoint=None, jump_nodes: bool = True,
                    centre_refine: bool = False):
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
    jump_nodes : (diffrax) stop steps at every EOS table node (default).  False
              is an opt-out for bulk value-only scans of tables with thousands
              of rows (also exposed by ``build_family``): cheaper (MetaModel
              family 3x), but Lambda loses ~1e3x in accuracy (2e-7 -> 2e-4..3e-3
              at the default rtol), the parameter tangents are biased by up to
              ~10%, and batched vs unbatched solves agree only to ~1e-4
              (measured 3e-5 in Lambda; 1e-12 with the default).  It is
              NOT the pre-2026-10-12 solver: the start stays frozen (so the h_c
              tangent stays accurate, 1-19% off the old one); parameter
              tangents equal the old solver's, and values equal it only up to
              step-decision roundoff (up to ~1e-4 in Lambda).
    centre_refine : (diffrax) integrate the inner 5% of (h_c - h_min) at
              rtol * 1e-3, floored at min(rtol, 1e-13) (never looser than the
              rest of the star).  Off by default.  Cuts the h_c-tangent error at the
              default rtol from ~1.2e-4 to ~8e-6 for +40-90% cost; parameter
              tangents and values are unchanged.

    Derivatives: forward-mode first derivatives in h_c and in any EOS
    parameter, and mixed d^2/dh_c dtheta, are supported.  Second derivatives
    in h_c (d^2 X/dh_c^2) are NOT: they are wrong by orders of magnitude.
    """
    if backend == "diffrax":
        return _solve_diffrax(table, h_c, rtol, atol, tidal, adjoint=adjoint,
                              jump_nodes=jump_nodes, centre_refine=centre_refine)
    elif backend == "scan":
        return _solve_scan(table, h_c, n_steps, tidal)
    raise ValueError(f"unknown backend {backend!r} (use 'diffrax' or 'scan')")
