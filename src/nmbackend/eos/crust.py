"""Tabulated SLy crust joined to a uniform-matter core.

Crust: the SLy unified EOS (BPS + Haensel-Pichon outer crust, Douchin &
Haensel 2001 inner crust, astro-ph/0111092), ``data/crust_sly_dh2001.dat``.
Below ``n_t`` the table is used as is.  This is the fixed crust of the paper-1
crust+cEFT ensemble (n_t = 0.055 fm^-3 = 0.34 n_sat).

Join (the P(rho) matching of Fortin et al. 2016, arXiv:1604.01944, Sec. II.B):

* The core table starts at the static density n_j = ``bridge`` * n_t (the
  models put a grid node exactly there).
* Join density n*: n_j if the core pressure there is at least
  p_t (eps_j/eps_t)^GAMMA_MIN, else the first density above n_j where the core
  reaches that curve (found by interpolation, so n* is continuous in the core
  parameters).  Core nodes in [n_j, 2 n_j) move affinely in ln n onto
  [n*, 2 n_j) and take the core values there; with n* = n_j nothing moves.
  Node count and order never change, so the table is continuous in the
  parameters.
* Between the crust point (eps_t, p_t) and the core at n*, p is a power law
  in eps (index >= GAMMA_MIN) on fixed segment nodes; p is continuous and
  strictly increasing.
* n follows the first law, d ln n = d eps/(eps + p), along the crust and the
  segment.  It then misses the core n* by a constant factor f (|f - 1| is
  1e-3 to 1e-2).  We rescale the crust and segment n by f (Fortin et al.
  Eq. 11, applied on the crust side).

* Rejection: a core that does not reach p_t (eps/eps_t)^GAMMA_MIN below
  2 n_j e^-0.1, or whose pressure is at the model floor (1e-9 MeV fm^-3) or not
  increasing anywhere between the join n* and 2 n_j, is outside the model's
  validity above the crust.  The table is then NaN (under jit) or InvalidCoreError is raised
  (eager); samplers treat both as failed draws.  Cores just inside this
  boundary (low L) join with slope kinks of up to ~6% in Lambda(L).

Unstable core (:func:`monotone_core`, :func:`finish_table`): the models floor
and cummax the core pressure, so where the raw pressure falls with density the
table has a plateau (cs2 = 1e-8, first-law residual up to ~4e-2).  A core whose
raw pressure falls by more than roundoff, or sits at the floor, above 2 n_j is
rejected in the same way when a star centred at the onset of that interval
lies below the maximum-mass star (UnstableCoreError when eager).  Accepted cores then carry no plateau
below the maximum-mass star, and Lambda approaches the rejection boundary
smoothly.  ``allow_unstable_core=True`` on the model keeps the plateau tables.

The TOV equations see only p(eps), which is the core's own above n_j; the
core baryon density, and hence the composition and dUrca onset density, are
untouched.  Vary ``n_t`` to estimate the crust-matching systematic; above
n_t = 0.076 fm^-3 (the SLy crust-core edge) the "crust" includes SLy4 core
rows of the table.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp

N_T_DEFAULT = 0.055       # fm^-3, 0.34 n_sat (paper 1, GKKS arXiv:2512.18044)
N_FLOOR = 3e-10           # fm^-3, lowest crust node: p = 4e-12 MeV fm^-3, chosen
                          # above the 1e-12 floor compose.eos_table_from_arrays
                          # had until 2026-10-25
BRIDGE = 1.25             # the core table starts at n_j = BRIDGE * n_t
GAMMA_MIN = 0.5           # the segment p ~ eps^Gamma has Gamma >= GAMMA_MIN
REMAP = 2.0               # core nodes in [n_j, REMAP n_j) may move (see join)
P_FLOOR = 1e-9            # MeV fm^-3, the pressure floor of the core models
JOIN_CAP = 0.1            # n* <= REMAP n_j e^-JOIN_CAP
P_DROP_RTOL = 1e-10       # pressure drops below this relative size are roundoff
M_LOW = 0.5               # Msun; below it M falls with h_c only below the M_min star
H_TOP = 0.92              # tov.default_hc_grid stops at H_TOP h_max
N_CRUST_PTS = 150
N_SEG_PTS = 16            # interior nodes of the power-law segment

_G_CM3_TO_MEVFM3 = 5.60958860e-13      # (1 g/cm^3) c^2 in MeV/fm^3
_DYN_CM2_TO_MEVFM3 = 6.241509074e-34
_TABLE = None


def sly_crust():
    """SLy crust table (n [fm^-3], eps, p [MeV/fm^3]) as NumPy arrays."""
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                            "crust_sly_dh2001.dat")
        n, rho, P = np.loadtxt(path, comments="#", unpack=True)
        _TABLE = (n, rho * _G_CM3_TO_MEVFM3, P * _DYN_CM2_TO_MEVFM3)
    return _TABLE


def crust_nodes(n_t=N_T_DEFAULT, n_floor=N_FLOOR, n_pts=N_CRUST_PTS):
    """Crust (n, eps, p) on a log grid from ``n_floor`` to ``n_t`` inclusive."""
    n0, e0, p0 = sly_crust()
    if not n0[0] <= n_floor < n_t <= n0[-1]:
        raise ValueError(f"n_t={n_t} outside the crust table")
    n = np.geomspace(n_floor, n_t, n_pts)
    ln = np.log(n)
    e = np.exp(np.interp(ln, np.log(n0), np.log(e0)))
    p = np.exp(np.interp(ln, np.log(n0), np.log(p0)))
    return n, e, p


def attach_crust(n, eps, p, n_t=N_T_DEFAULT, n_floor=N_FLOOR,
                 n_crust_pts=N_CRUST_PTS, n_seg=N_SEG_PTS, n_quad=257,
                 gamma_min=GAMMA_MIN):
    """Join the SLy crust below ``n_t`` to a core (n, eps, p) whose first node
    is n_j = BRIDGE * n_t (fm^-3, MeV/fm^3).  JAX-traceable in the core arrays;
    the node count is static.  Returns the combined (n, eps, p)."""
    n, eps, p = jnp.asarray(n), jnp.asarray(eps), jnp.asarray(p)
    nc, ec, pc = crust_nodes(n_t, n_floor, n_crust_pts)
    et, pt = ec[-1], pc[-1]
    ln, le, lp = jnp.log(n), jnp.log(eps), jnp.log(p)
    lnj = math.log(join_density(n_t))
    lnw = lnj + math.log(REMAP)
    bad = join_invalid(n, eps, p, n_t, gamma_min)
    # join density n*: n_j if the core there lies on or above
    # p_t (eps/eps_t)^gamma_min, else the first crossing of that curve
    # (linear in ln n between core nodes, so continuous in the parameters)
    lnx = jnp.minimum(_join_lnn(ln, le, lp, et, pt, gamma_min), lnw - JOIN_CAP)
    # core nodes in [n_j, REMAP n_j) move affinely (in ln n) onto [n*, REMAP n_j)
    # and take the core values there (log-log interpolation); the identity when
    # n* = n_j, and continuous in n* otherwise
    inw = ln < lnw
    lm = jnp.where(inw, lnx + (ln - ln[0]) * (lnw - lnx) / (lnw - ln[0]), ln)
    n = jnp.where(inw, jnp.exp(lm), n)
    eps = jnp.where(inw, jnp.exp(jnp.interp(lm, ln, le)), eps)
    p = jnp.where(inw, jnp.exp(jnp.interp(lm, ln, lp)), p)
    # power law p(eps) from the crust point to the (moved) first core node
    lr = jnp.log(eps[0] / et)
    gam = jnp.log(p[0] / pt) / lr
    # ln n - ln n_t = int deps/(eps + p) along the segment, on a fine s grid
    sq = jnp.linspace(0.0, 1.0, n_quad)
    eq = et * jnp.exp(sq * lr)
    integrand = eq / (eq + pt * (eq / et) ** gam) * lr               # d/ds
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(sq))])
    f = n[0] / (n_t * jnp.exp(cum[-1]))          # rescales crust + segment n
    s = jnp.linspace(0.0, 1.0, n_seg + 2)[1:-1]
    es = et * jnp.exp(s * lr)
    ns = f * n_t * jnp.exp(jnp.interp(s, sq, cum))
    ps = pt * (es / et) ** gam
    nan = jnp.where(bad, jnp.nan, 0.0)          # invalid core: NaN table
    return (jnp.concatenate([f * jnp.asarray(nc), ns, n]) + nan,
            jnp.concatenate([jnp.asarray(ec), es, eps]) + nan,
            jnp.concatenate([jnp.asarray(pc), ps, p]) + nan)


class InvalidCoreError(ValueError):
    """The uniform-matter core cannot be joined to the crust (see
    :func:`join_invalid`); the draw is outside the model's validity."""


class UnstableCoreError(InvalidCoreError):
    """The core's raw pressure falls with density below the maximum-mass star
    (see :func:`monotone_core`).  Under jit such a table is NaN like any other
    rejected core; an eager rebuild tells the two apart."""


def _join_lnn(ln, le, lp, et, pt, gamma_min):
    """ln n*: ln n_j if the core there lies on or above p_t (eps/eps_t)^gamma_min,
    else the first crossing of that curve (linear in ln n between nodes, so
    continuous in the parameters); +inf if there is none."""
    g = lp - math.log(pt) - gamma_min * (le - math.log(et))
    up = (ln > ln[0]) & (g >= 0.0)
    k = jnp.maximum(jnp.argmax(up), 1)
    t = jnp.clip(-g[k - 1] / jnp.where(g[k] > g[k - 1], g[k] - g[k - 1], 1.0), 0.0, 1.0)
    lnx = jnp.where(g[0] >= 0.0, ln[0], ln[k - 1] + t * (ln[k] - ln[k - 1]))
    return jnp.where((g[0] >= 0.0) | jnp.any(up), lnx, jnp.inf)


def join_invalid(n, eps, p, n_t=N_T_DEFAULT, gamma_min=GAMMA_MIN):
    """True if :func:`attach_crust` must reject this core (see
    :class:`InvalidCoreError`).  JAX-traceable; ``n`` starts at n_j.

    Rejected: the join density n* (where the core first reaches
    p_t (eps/eps_t)^gamma_min) lies beyond the cap 2 n_j e^-0.1, or the core
    pressure is at the floor or not increasing anywhere between n* and 2 n_j.
    Below n* the extended crust replaces the core, so a floor there is allowed:
    low-L cores are mechanically unstable just above n_t, where the real crust
    extends further."""
    n, eps, p = jnp.asarray(n), jnp.asarray(eps), jnp.asarray(p)
    _, ec, pc = crust_nodes(n_t)
    et, pt = ec[-1], pc[-1]
    ln = jnp.log(n)
    lnw = math.log(join_density(n_t) * REMAP)
    lnx = _join_lnn(ln, jnp.log(eps), jnp.log(p), et, pt, gamma_min)
    above = (ln < lnw) & (ln >= lnx - 1e-12)
    flat = jnp.any(above[1:] & above[:-1] & (jnp.diff(p) <= 0.0))
    floor = jnp.any(above & (p <= P_FLOOR * 1.001))
    return flat | floor | (lnx > lnw - JOIN_CAP)


def join_density(n_t=N_T_DEFAULT):
    """n_j, where the core table starts (fm^-3)."""
    return BRIDGE * n_t


def core_grid(n_grid, n_t=N_T_DEFAULT):
    """Static core grid for the crust join: n_j, then the nodes of ``n_grid``
    above it (a node closer than 1% to n_j is dropped)."""
    nj = join_density(n_t)
    g = np.asarray(n_grid)
    return jnp.asarray(np.concatenate([[nj], g[g > nj * 1.01]]))


CRUSTS = ("sly", "polytrope")


def check_crust(crust, n_surface=None):
    """Validate ``crust``; warn that a non-default ``n_surface`` only acts with
    the polytrope (``n_surface=None`` still disables the low-density part)."""
    if crust not in CRUSTS:
        raise ValueError(f"crust={crust!r}; use one of {CRUSTS}")
    if crust == "sly" and n_surface is not None and n_surface != 1e-7:
        warnings.warn("n_surface is ignored with crust='sly' (the SLy table "
                      f"starts at {N_FLOOR} fm^-3)", stacklevel=3)
    return crust


def monotone_core(n, p, n_t=N_T_DEFAULT):
    """Floor and cummax the raw core pressure ``p`` (MeV fm^-3) on ``n``
    (fm^-3).  Returns (p_monotone, ln n_u): n_u is where the first unstable
    interval above REMAP n_j begins, +inf if there is none.  An interval is a
    run of nodes where the raw p is at the floor or more than P_DROP_RTOL below
    the running maximum; it begins at the node that set the maximum.
    JAX-traceable."""
    n, p = jnp.asarray(n), jnp.asarray(p)
    pf = jnp.maximum(p, P_FLOOR)
    pm = jax.lax.cummax(pf)
    ln = jnp.log(n)
    drop = (pf < pm * (1.0 - P_DROP_RTOL)) | (pf <= P_FLOOR * 1.001)
    idx = jnp.arange(n.shape[0])
    anchor = jax.lax.cummax(jnp.where(drop, 0, idx))    # last node not in a drop
    above = drop & (ln >= math.log(REMAP * join_density(n_t)))
    return pm, jnp.min(jnp.where(above, ln[anchor], jnp.inf))


def unstable_below_mmax(table, ln_nu):
    """True if a star centred at n_u lies below the maximum-mass star: M rises
    with h_c there, or M < M_LOW (the falling branch below the minimum-mass
    star).  Onsets above H_TOP h_max, beyond the stellar sequences of
    :func:`nmbackend.tov.default_hc_grid`, are kept.  Solves one star, only
    where needed (under vmap it solves a harmless star at h_max/2 instead).
    JAX-traceable."""
    from ..tov.solver import solve_structure

    def check(t, hu):
        m, dm = jax.jvp(lambda h: solve_structure(t, h).M_msun, (hu,),
                        (jnp.ones_like(hu),))
        return (dm > 0.0) | (m < M_LOW)

    hmax = table.h[-1]
    hu = jnp.interp(jnp.minimum(ln_nu, jnp.log(table.n[-1])),
                    jnp.log(table.n), table.h)
    ok = (jnp.isfinite(ln_nu) & (hu < H_TOP * hmax)
          & jnp.all(jnp.isfinite(table.p)))
    hu = jnp.where(ok, hu, 0.5 * hmax)
    try:                                          # eager: skip the solve if stable
        if not bool(ok):
            return jnp.asarray(False)
        return check(table, hu)
    except jax.errors.ConcretizationTypeError:
        return jax.lax.cond(ok, check, lambda t, h: jnp.asarray(False),
                            table, hu)


def _raise_if(bad, err, msg):
    try:                                          # eager call: raise; jit: NaN table
        if bool(bad):
            raise err(msg)
    except jax.errors.ConcretizationTypeError:
        pass


def finish_table(n, eps, p, crust, n_t, n_surface, gamma_crust,
                 ln_nu=math.inf, allow_unstable=False):
    """Low-density completion shared by the uniform-matter models, then the
    EOSTable.  ``crust="sly"``: :func:`attach_crust` (the core must start at
    :func:`join_density`).  ``crust="polytrope"``: the legacy surface continuation
    :func:`nmbackend.eos.base.extend_to_surface`.  ``n_surface=None`` skips
    both (table truncated at the core grid start).

    ``ln_nu`` is the unstable-interval onset from :func:`monotone_core`.  Unless
    ``allow_unstable``, the table is rejected (InvalidCoreError when eager, NaN
    under jit) if :func:`unstable_below_mmax`."""
    return raise_rejected(*finish_table_flags(
        n, eps, p, crust, n_t, n_surface, gamma_crust, ln_nu, allow_unstable))


def finish_table_flags(n, eps, p, crust, n_t, n_surface, gamma_crust,
                       ln_nu=math.inf, allow_unstable=False):
    """:func:`finish_table` without the raise, for use inside jit: returns
    (table, bad_join, bad_unstable), with a NaN table if either flag is set.
    :func:`raise_rejected` then raises when the flags are concrete."""
    from .base import EOSTable, build_h_and_cs2, extend_to_surface
    bad_join = jnp.asarray(False)
    if n_surface is not None:
        if crust == "sly":
            n, eps, p = attach_crust(n, eps, p, n_t=n_t)
            bad_join = ~jnp.all(jnp.isfinite(p))
        else:
            n, eps, p = extend_to_surface(n, eps, p, n_surface=n_surface,
                                          gamma_crust=gamma_crust)
    table = build_h_and_cs2(n, eps, p)
    if allow_unstable:
        return table, bad_join, jnp.asarray(False)
    bad = unstable_below_mmax(table, ln_nu)
    return EOSTable(*(jnp.where(bad, jnp.nan, a) for a in table)), bad_join, bad


def raise_rejected(table, bad_join, bad_unstable):
    """Raise InvalidCoreError / UnstableCoreError for concrete flags from
    :func:`finish_table_flags`; under a trace, return the (NaN) table."""
    _raise_if(bad_join, InvalidCoreError, "core invalid above the crust (p at "
                                          "the floor or not increasing near n_j)")
    _raise_if(bad_unstable, UnstableCoreError, "core mechanically unstable "
                                               "(dp/dn <= 0) below the "
                                               "maximum-mass central density")
    return table
