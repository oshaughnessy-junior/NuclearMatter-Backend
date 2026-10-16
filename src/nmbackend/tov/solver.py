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

  The tidal term (eps + p)/cs2 is the analytic deps/dh of the same Hermite cell
  that gives eps(h) (``EOSTable.at_h_tidal``), so per cell it integrates to
  exactly the Delta eps the structure sees (docs/devlog/2026-10-14-tidal-cs2.md).
  Before 2026-10-14 it took the node cs2 interpolated linearly in h, which left
  a "phantom" Delta eps at steep joints (Lambda up to ~10% low).  The new term
  jumps at the nodes, so y is carried in a node-anchored picture
  (:func:`_derivs_x`; equal to h in value) in which the nodes do not move with
  the EOS parameters, and the frozen mesh keeps the jumps' boundary terms in
  the parameter tangents.  First parameter derivatives of Lambda only.

* ``backend="scan"``: fixed-step RK4 in a ``jax.lax.scan``, the reverse-mode
  path (``jax.grad`` / ``jax.jacrev``, with or without the tidal sector).  It
  steps in sigma = sqrt((h_c - h) / (h_c - h_min)), in which the centre is
  regular, on a frozen mesh: sin-mapped base steps, every table node, and
  points where the EOS varies fastest (``_scan_mesh``).  y is carried in a
  node-anchored picture like the diffrax path's.  Measured at the default
  n_steps=2000, against the diffrax backend:

  - values: ~1e-5 in M and R, ~5e-5 in Lambda (worst at 0.26 Msun); error
    falls as n_steps^-4;
  - first derivatives (forward or reverse mode): ~6e-5; error falls as about
    n_steps^-3.5 above 2000 steps, slower below;
  - the AD derivative equals the finite difference of the scan's own values
    to ~2e-7 (the mesh is frozen; FD steps that move a node across a mesh
    point see small kinks);
  - SECOND derivatives are NOT supported: O(1) wrong at any n_steps (see
    solve_structure).

  Cost: at least len(table.h) steps whatever n_steps is (every node is a
  mesh point), and a tidal reverse-mode gradient costs ~16x a value solve.
  Before 2026-10-14 it stepped uniformly in h and was wrong at every h_c, with
  M off by up to +210% and Lambda NaN
  (docs/devlog/2026-10-14-tov-scan-accuracy.md).

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
    import equinox as eqx
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



def _rhs(eps, p, deps_dh, state, tidal):
    """d(r, m, y)/dh from the EOS at one point.  ``deps_dh`` is the tidal term
    (eps + p)/cs^2, taken as the analytic deps/dh of the Hermite cell that also
    gives eps, so the tidal source integrates to exactly the structure's
    Delta eps per cell (docs/devlog/2026-10-14-tidal-cs2.md)."""
    r, m, y = state
    denom = m + _FOURPI * r**3 * p
    drdh = -r * (r - 2.0 * m) / denom
    dmdh = _FOURPI * r**2 * eps * drdh

    if tidal:
        one_minus_2mr = 1.0 - 2.0 * m / r
        F = (1.0 - _FOURPI * r**2 * (eps - p)) / one_minus_2mr
        Q = (_FOURPI / one_minus_2mr) * (
            5.0 * eps + 9.0 * p + deps_dh - 6.0 / (_FOURPI * r**2)
        ) - (2.0 * (m + _FOURPI * r**3 * p) / (r**2 * one_minus_2mr)) ** 2
        dydr = -(y**2 + y * F + r**2 * Q) / r
        dydh = dydr * drdh
    else:
        # structure-only: keep y constant so its singular near-centre terms are
        # never formed -> reverse-mode AD of (R, M) is clean.
        dydh = 0.0 * r
    return jnp.stack([drdh, dmdh, dydh])


# ----- the tidal variable in a node-anchored picture (adaptive backend) -----
# The tidal term deps/dh jumps at every table node (the Hermite of p(h) is C^1).
# The adaptive solve stops at the nodes with a mesh frozen against
# differentiation, so a node whose h moves with the EOS parameters carries a
# boundary term (f_above - f_below) dh_node/dtheta that the frozen mesh never
# sees (measured: dLambda/dL_sym off by a factor 2.6, dLambda/dscale on the
# steep-joint table by 2.5).  So y is carried in a node-anchored picture: at
# solve coordinate x, y holds y(h(x)), where h(x) is piecewise linear with knots
# at the nodes' gradient-stopped positions U = stop_gradient(table.h) <-> h =
# table.h.  The nodes then sit at fixed x for y, and their motion enters through
# h(x) and dh/dx.  r and m stay in h (x == h): their right-hand side is
# continuous at the nodes, so they need no boundary term, and moving them too
# costs accuracy (dM/dL_sym 1e-6 -> 6e-4 at the default rtol, from the large
# intermediate shift tangent).  The y equation needs r, m at h(x); they are
# taken to first order, r + (dr/dh) (h(x) - x), exact for first tangents since
# h(x) - x is zero in value.  The cell holding the (frozen) start ``ua`` is
# mapped onto [h_{i-1}, ua], so h(x) = x at the start; at the end of a solve
# y is moved back to h = x by y - (dy/dx)(h(x) - x).  At the primal U == table.h,
# so every value is bit-identical to integrating y in h; only first tangents
# change, by exactly the boundary terms.  Second derivatives in EOS parameters
# (d^2/dtheta^2) were never supported and also miss the O((h(x) - x)^2) terms.
def _u_cell(table: EOSTable, x, ua):
    """Cell index and the y-picture map of cell i: (lo_u, lo_h, dh/dx) at ``x``,
    with the cell holding the start ``ua`` anchored there."""
    H = table.h
    U = jax.lax.stop_gradient(H)
    i = jnp.clip(jnp.searchsorted(U, x), 1, H.shape[0] - 1)
    top = U[i] >= ua                              # the cell holding the start
    hi_u = jnp.where(top, ua, U[i])
    hi_h = jnp.where(top, ua, H[i])
    return i, U[i - 1], H[i - 1], (hi_h - H[i - 1]) / (hi_u - U[i - 1])


def _shift(x, lo_u, lo_h, dhdx):
    """h(x) - x of the y picture; written so that it is exactly 0 in value."""
    return (lo_h - lo_u) + (x - lo_u) * (dhdx - 1.0)


def _cell_eos(table: EOSTable, i, t, dh, aux, tidal: bool):
    """(eps, p, tidal term) in table cell ``i`` (nodes i-1, i; h-width ``dh``)
    at local ``t`` (extrapolated outside [0, 1]); the tidal term is 0.0 unless
    ``tidal``.  ``aux`` is ``eos.base._degenerate_cells(table.h, table.eps,
    table.p)``.  Shared by the adaptive (:func:`_derivs_x`) and scan backends."""
    from ..eos.base import _floor_eps_p, _hermite_cell_t, _tidal_term
    if not tidal:
        eps, p = _hermite_cell_t(i, t, dh, table.p, table.eps)
        eps, p = _floor_eps_p(eps, p, jnp)
        return eps, p, 0.0
    eps, p, deps = _hermite_cell_t(i, t, dh, table.p, table.eps, deps=True)
    eps, p = _floor_eps_p(eps, p, jnp)
    return eps, p, _tidal_term(i, t, dh, deps, eps, p, *aux, jnp)


def _derivs_x(x, state, table: EOSTable, ua, aux, tidal: bool = True):
    """d(r, m, y)/dx (adaptive backend): r, m in h (x == h), y in the
    node-anchored picture (see the note above).  ``aux`` is
    ``eos.base._degenerate_cells(table.h, table.eps, table.p)``."""
    i, lo_u, lo_h, dhdx = _u_cell(table, x, ua)
    dh = table.h[i] - lo_h
    eps, p, _ = _cell_eos(table, i, (x - lo_h) / dh, dh, aux, False)
    f = _rhs(eps, p, 0.0, state, False)           # (dr/dh, dm/dh, 0)
    if not tidal:
        return f
    t_y = (x - lo_u) * dhdx / dh                  # == (h(x) - h_{i-1}) / dh
    eps_y, p_y, deps_y = _cell_eos(table, i, t_y, dh, aux, True)
    s_y = state + f * _shift(x, lo_u, lo_h, dhdx)  # r, m at h(x); f[2] == 0
    fy = _rhs(eps_y, p_y, deps_y, s_y, True)[2] * dhdx
    return jnp.stack([f[0], f[1], fy])


def _y_to_h(table, x, state, ua, aux):
    """Tidal state at solve coordinate ``x`` with y moved from h(x) back to h = x
    (a zero-valued correction; first tangents exact)."""
    _, lo_u, lo_h, dhdx = _u_cell(table, x, ua)
    fy = _derivs_x(x, state, table, ua, aux, True)[2]
    return state.at[2].add(-fy * _shift(x, lo_u, lo_h, dhdx))


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
# docs/devlog/2026-10-14-tov-scan-accuracy.md.  Until 2026-10-14 the scan stepped
# uniformly in h from h_c - dh0; with r ~ sqrt(h_c - h) its first step
# overshot r by orders of magnitude: M +60..+210% off and Lambda NaN at almost
# every h_c at the default n_steps, on every table.
#
# Now it steps in sigma = sqrt((h_c - h) / H), H = h_c - h_min, in which r is
# regular at the centre, on a mesh frozen against differentiation
# (:func:`_scan_mesh`) whose primal points include every table node.  Each step
# evaluates the EOS on its own table cell.  As in the diffrax path, r and m are
# carried on the frozen mesh: their right-hand side is continuous at the nodes,
# so the node motion enters through the vector field.  The tidal term jumps at
# the nodes, so y is carried in a node-anchored picture (cf. :func:`_derivs_x`):
# in step cell i, y holds y(sigma~(sigma)), with sigma~ linear in the cell and
# mapping its frozen ends to their live positions.  The jump then contributes
# its boundary term to the tangent.  sigma~ == sigma in value, and it equals
# sigma at both ends of the star (sigma0 and the surface), so no start or end
# correction is needed.
def _scan_mesh(table, h_c, dh0, n_steps, aux):
    """Frozen sigma mesh, increasing from sigma0 = sqrt(dh0 / H) to 1 (static
    length n_steps + n_steps//4 + len(table.h)), and the table cell of each step.

    * ``n_steps + 1`` base points equally spaced in s, at sigma = sigma0 +
      (1 - sigma0) sin(pi s / 2): the sqrt resolves the centre (r ~ sqrt(h_c -
      h)), the sin packs steps toward the surface (with uniform sigma the Lambda
      error stalls near 4e-6);
    * every table node, so no step straddles a node, where eps' and the tidal
      term jump (jump cells of ``_degenerate_cells`` included);
    * ``n_steps // 4`` points where the EOS varies fastest (:func:`_eos_points`).
      Where the table is denser than the base grid a step is one cell however
      large n_steps is; at the MetaModel crust-core joint the Lambda tangent was
      then up to 9e-3 off at n_steps=2000 (values 1e-6).

    Points outside the star clip to an end and give zero-length steps.  The mesh
    is frozen (like the diffrax ``jump_ts``): a mesh that moves with the nodes
    either rescales the near-centre steps with the node just below h_c
    (dM/dL_sym 6e-5 -> 1.2e-2 off) or lets base points cross nodes (a
    piecewise-smooth map whose Lambda ripples in the parameters)."""
    sg = jax.lax.stop_gradient
    tb = jax.tree_util.tree_map(sg, table)
    aux = jax.tree_util.tree_map(sg, aux)
    hn, N = tb.h, tb.h.shape[0]
    hc, top = sg(h_c), sg(h_c - dh0)
    H = hc - hn[0]
    s0 = jnp.sqrt(sg(dh0) / H)
    s = jnp.linspace(0.0, 1.0, n_steps + 1)
    base = s0 + (1.0 - s0) * jnp.sin(0.5 * math.pi * s)
    base = base.at[0].set(s0).at[-1].set(1.0)
    h_pts = jnp.concatenate([hn[1:], _eos_points(tb, top, n_steps // 4, aux)])
    pts = jnp.sqrt(jnp.clip((hc - h_pts) / H, s0 * s0, 1.0))
    sig = jnp.sort(jnp.concatenate([base, pts]))
    mid = 0.5 * (sig[:-1] + sig[1:])
    cell = jnp.clip(jnp.searchsorted(hn, hc - H * mid * mid), 1, N - 1)
    return sig, cell


def _eos_points(table, top, k, aux):
    """``k`` enthalpies in [h_min, top], equidistributed in the EOS variation per
    cell: |dln cs2| + |dln eps| (node values) in a regular cell, nothing in a
    jump cell, and ln(1 + Delta eps / eps) of the jump run above in the cell
    that carries its junction bubble (:func:`eos.base._tidal_term`), whose
    points go to t = u^(1/(q+1)), equidistributing the bubble (q+1) t^q.
    Call with gradient-stopped inputs."""
    from ..eos.base import _BUBBLE_Q
    h, eps, cs2 = table.h, table.eps, table.cs2
    deg, extra = aux[0][1:], aux[1][1:]                    # per cell 1..N-1
    width = h[1:] - h[:-1]
    lo, hi = jnp.clip(h[:-1], h[0], top), jnp.clip(h[1:], h[0], top)
    frac = jnp.where(width > 0, (hi - lo) / jnp.where(width > 0, width, 1.0), 0.0)
    w_reg = jnp.abs(jnp.diff(jnp.log(cs2))) + jnp.abs(jnp.diff(jnp.log(eps)))
    w_bub = jnp.log1p(extra / eps[1:])
    w = (jnp.where(deg, 0.0, w_reg) + w_bub + 1e-12) * frac
    W = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w)])
    q = (jnp.arange(k) + 0.5) / k * W[-1]
    j = jnp.clip(jnp.searchsorted(W, q), 1, h.shape[0] - 1)
    u = (q - W[j - 1]) / (W[j] - W[j - 1])
    u = jnp.where(extra[j - 1] > 0, u ** (1.0 / (_BUBBLE_Q + 1.0)), u)
    return lo[j - 1] + u * (hi[j - 1] - lo[j - 1])


def _solve_scan(table, h_c, n_steps, tidal):
    from ..eos.base import _degenerate_cells
    sg = jax.lax.stop_gradient
    h_min, h_c, dh0, state0 = _central_start(table, h_c)
    aux = _degenerate_cells(table.h, table.eps, table.p, jnp)
    sig, cell = _scan_mesh(table, h_c, dh0, n_steps, aux)
    H = h_c - h_min
    lo_h = table.h[cell - 1]
    dh = table.h[cell] - lo_h
    xs = (sig[:-1], jnp.diff(sig), cell, lo_h, dh)
    if tidal:
        # y picture per step: the cell's ends in sigma, live (a, b) and frozen
        # (A, B); the top cell ends at the start sigma0, the bottom one at 1
        s0 = jnp.sqrt(dh0 / H)
        top = table.h[cell] >= h_c - dh0
        a = jnp.where(top, s0,
                      jnp.sqrt(jnp.where(top, 1.0, (h_c - table.h[cell]) / H)))
        b = jnp.sqrt((h_c - lo_h) / H)
        A, B = sg(a), sg(b)
        g = (b - a) / jnp.where(B > A, B - A, 1.0)           # d sigma~/d sigma
        xs = xs + (a, A, g)

    def step(state, sd):
        s_lo, ds, i, lo, w = sd[:5]

        def f(s, X):            # d(r, m, y)/dsigma
            eps, p, _ = _cell_eos(table, i, (h_c - H * s * s - lo) / w, w, aux,
                                  False)
            frm = _rhs(eps, p, 0.0, X, False) * (-2.0 * H * s)  # (dr, dm, 0)
            if not tidal:
                return frm
            a, A, g = sd[5:]
            shift = _shift(s, A, a, g)                      # sigma~ - sigma == 0
            st = s + shift
            eps, p, d = _cell_eos(table, i, (h_c - H * st * st - lo) / w, w, aux,
                                  True)
            fy = _rhs(eps, p, d, X + frm * shift, True)[2] * (-2.0 * H * st) * g
            return frm.at[2].set(fy)

        k1 = f(s_lo, state)
        k2 = f(s_lo + 0.5 * ds, state + 0.5 * ds * k1)
        k3 = f(s_lo + 0.5 * ds, state + 0.5 * ds * k2)
        k4 = f(s_lo + ds, state + ds * k3)
        return state + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    state, _ = jax.lax.scan(step, state0, xs)
    return _finish(state[0], state[1], state[2], h_c, tidal)


# ----- adaptive diffrax backend -----
# Module-level vector fields: diffeqsolve is jitted with the term as a static key, so a
# fresh closure per call would recompile every solve and leak the executables.
def _field_tidal(x, s, args):
    return _derivs_x(x, s, *args, True)          # args = (table, ua, aux)


def _field_structure(x, s, args):
    return _derivs_x(x, s, *args, False)


def _node_jumps(table):
    """Interior EOS-table nodes as diffrax ``jump_ts`` (gradient-stopped).

    The TOV right-hand side is not smooth at any table node: eps comes from a
    C^1 Hermite of p(h) (so eps' jumps), and the tidal term deps/dh of the same
    Hermite jumps.  An adaptive step that straddles a node rejects repeatedly
    there, and ForwardMode then differentiates the mesh's position relative to
    the non-smooth point -- a tangent bias that shrinks only slowly and
    erratically with rtol (docs/devlog/2026-10-12-tov-jump-ts.md).  Stepping to
    just before and restarting just after each node removes both.
    ``table.h[0]`` is the surface (the integration end point) and is left out;
    nodes outside the integration interval are inert, so the array keeps the
    table's static length under jit/vmap.  The positions are gradient-stopped:
    the mesh is fixed.  For r and m (continuous right-hand side) the node motion
    enters through the vector field on each side of the node; for y (which
    jumps) through its node-anchored picture (:func:`_derivs_x`), which carries
    the boundary term (devlog 2026-10-14-tidal-cs2)."""
    return jax.lax.stop_gradient(table.h[1:])


def _adaptive_tov(table, t0, t1, state0, rtol, atol, tidal, dt0,
                  adjoint=None, max_steps=100000, saveat=None, jump_nodes=True):
    """Adaptive Dopri5 integration of the TOV(+tidal) system from ``t0`` down to
    ``t1`` in pseudo-enthalpy, with steps cut at the EOS table nodes.

    Three mechanisms, each fixing part of the tangent (docs/devlog/
    2026-10-12-tov-jump-ts.md, 2026-10-14-tidal-cs2.md):

    * ``jump_ts`` at the table nodes fixes the *parameter* tangents: the mesh
      no longer straddles kinks whose position moves with the EOS parameters.
    * The tidal y is carried in a node-anchored picture (:func:`_derivs_x`), so
      the jump of the tidal term deps/dh at a node that moves with the
      parameters contributes its boundary term, which the frozen mesh would
      drop.  The returned y is moved back to h = t1 (:func:`_y_to_h`).
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
    mixed d^2/dh_c dtheta and first parameter derivatives are fine; second
    parameter derivatives are not (see solve_structure).
    ``tests/test_tov_jump_ts.py::test_second_hc_derivative_is_not_supported``
    pins this.  With ``tidal=True`` only the default ``saveat`` (the end
    point) is supported: saved y would be in the node-anchored picture."""
    sg = jax.lax.stop_gradient
    t0, t1 = jnp.asarray(t0), jnp.asarray(t1)   # a Python float here is a jit key
    if tidal and saveat is not None:
        raise ValueError("tidal solves save the end point only")
    ua = sg(t0)                  # y picture anchored at the frozen start
    from ..eos.base import _degenerate_cells
    aux = _degenerate_cells(table.h, table.eps, table.p, jnp)   # jumps at fixed p
    t0s = sg(t0)                                # REQUIRED with jump_ts; see above
    f0 = _derivs_x(t0, state0, table, ua, aux, tidal)
    _, fdot = jax.jvp(lambda t, y: _derivs_x(t, y, table, ua, aux, tidal),
                      (t0, state0), (jnp.ones_like(t0), f0))
    d = t0s - t0
    state0 = state0 + f0 * d + 0.5 * fdot * d * d
    term = diffrax.ODETerm(_field_tidal if tidal else _field_structure)
    ctrl = diffrax.PIDController(
        rtol=rtol, atol=atol,
        jump_ts=_node_jumps(table) if jump_nodes else None)
    if adjoint is None:
        adjoint = diffrax.ForwardMode()   # supports values, vmap, forward-mode AD
    sol = diffrax.diffeqsolve(
        term, diffrax.Dopri5(), t0=t0s, t1=t1, dt0=sg(dt0),
        y0=state0, args=(table, ua, aux), stepsize_controller=ctrl, max_steps=max_steps,
        adjoint=adjoint,
        saveat=diffrax.SaveAt(t1=True) if saveat is None else saveat,
    )
    if not tidal:
        return sol
    ys = _y_to_h(table, t1, sol.ys[0], ua, aux)[None]
    return eqx.tree_at(lambda s: s.ys, sol, ys)


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
              "scan" (fixed-step RK4 on a frozen mesh; forward- and
              reverse-mode AD, with or without the tidal sector).
    tidal   : integrate the tidal sector (k2, Lambda).  ``tidal=False`` drops y
              (cheaper; k2 and Lambda are NaN placeholders).
    rtol, atol : diffrax tolerances.
    n_steps : scan-backend base step count (>= 1); the scan takes
              n_steps + n_steps // 4 + len(table.h) - 1 steps (table nodes and
              EOS-placed points, ``_scan_mesh``), so its cost never falls below
              len(table.h) steps.  At 2000: ~1e-5 in M, R and ~5e-5 in Lambda
              (worst at 0.26 Msun), first derivatives ~6e-5.
    adjoint : optional diffrax adjoint override (default ForwardMode()).
    jump_nodes : (diffrax) stop steps at every EOS table node (default).  False
              was an opt-out for bulk value-only scans of tables with thousands
              of rows (also exposed by ``build_family``).  Since 2026-10-14 the
              tidal term jumps at the nodes, so with tidal=True it no longer
              saves time (MetaModel family 81 vs 88 ms, GM1 78 vs 38 ms) and
              Lambda is ~1e4x less accurate (7e-4 MetaModel, 6e-3 GM1 at the
              default rtol, vs 2e-7); NOT recommended with tidal=True.  With
              tidal=False it is unaffected.  The parameter tangents are biased
              by up to ~10%, and batched vs unbatched solves agree only to ~1e-4
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
    parameter, and mixed d^2/dh_c dtheta, are supported.  With
    backend="scan", first derivatives in either mode are supported, and NO
    second derivatives (jax.hessian, jacfwd(jacrev)): the frozen mesh and the
    first-order y picture leave them O(1) wrong at any n_steps (MetaModel,
    h_c = 0.2: d^2(M, R, Lambda)/dh_c^2 off by -0.49, +0.93, -0.10; pinned by
    tests/test_tov_scan.py).  With the diffrax backend, second derivatives
    in h_c (d^2 X/dh_c^2) are NOT supported: they are wrong by orders of
    magnitude.
    Nor are second derivatives in EOS parameters (d^2/dtheta^2): on the
    MetaModel star of tests/test_tov_jump_ts.py, d^2(M, R, Lambda)/dL_sym^2 are
    off by O(1) (and were off by 5, 10 and 900x before 2026-10-14); the tidal
    y is moreover carried only to first order in the node motion (devlog
    2026-10-14-tidal-cs2).
    """
    if backend == "diffrax":
        return _solve_diffrax(table, h_c, rtol, atol, tidal, adjoint=adjoint,
                              jump_nodes=jump_nodes, centre_refine=centre_refine)
    elif backend == "scan":
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be at least 1 (got {n_steps})")
        return _solve_scan(table, h_c, int(n_steps), tidal)
    raise ValueError(f"unknown backend {backend!r} (use 'diffrax' or 'scan')")
