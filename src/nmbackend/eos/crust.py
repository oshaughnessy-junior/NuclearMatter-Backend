"""Tabulated SLy crust joined to a uniform-matter core.

Crust: the SLy unified EOS (BPS + Haensel-Pichon outer crust, Douchin &
Haensel 2001 inner crust, astro-ph/0111092), ``data/crust_sly_dh2001.dat``.
Below ``n_t`` the table is used as is.  This is the fixed crust of the paper-1
crust+cEFT ensemble (n_t = 0.055 fm^-3 = 0.34 n_sat).

Join (the P(rho) matching of Fortin et al. 2016, arXiv:1604.01944, Sec. II.B):

* The core table starts at the static density n_j = ``bridge`` * n_t (the
  models put a grid node exactly there) and is kept unchanged.
* Between the crust point (eps_t, p_t) and the core point (eps_j, p_j), p is
  a power law in eps on a fixed set of segment nodes, so p is continuous and
  strictly increasing in eps whenever p_j > p_t.  Node positions do not move
  with the core parameters, so the table is smooth in them.
* n follows the first law, d ln n = d eps/(eps + p), along the crust and the
  segment.  It then misses the core n_j by a constant factor f (|f - 1| is
  1e-3 to 1e-2).  We rescale the crust and segment n by f (Fortin et al.
  Eq. 11, applied on the crust side).
* Fallback, if p_j <= p_t (a core softer than the crust above n_t): the
  segment ends at the first core node with p > p_t (eps/eps_t), and the core
  nodes below it move onto the segment.  This branch is not smooth in the
  core parameters.

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
import jax.numpy as jnp

N_T_DEFAULT = 0.055       # fm^-3, 0.34 n_sat (paper 1, GKKS arXiv:2512.18044)
N_FLOOR = 3e-10           # fm^-3, lowest crust node: p = 4e-12 MeV fm^-3, above
                          # the 1e-12 floor of compose.eos_table_from_arrays
BRIDGE = 1.25             # the core table starts at n_j = BRIDGE * n_t
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
                 n_crust_pts=N_CRUST_PTS, n_seg=N_SEG_PTS, n_quad=257):
    """Join the SLy crust below ``n_t`` to a core (n, eps, p) whose first node
    is n_j = BRIDGE * n_t (fm^-3, MeV/fm^3).  JAX-traceable in the core arrays;
    the node count is static.  Returns the combined (n, eps, p)."""
    n, eps, p = jnp.asarray(n), jnp.asarray(eps), jnp.asarray(p)
    nc, ec, pc = crust_nodes(n_t, n_floor, n_crust_pts)
    et, pt = ec[-1], pc[-1]
    # segment end: the first core node, or (fallback) the first with p > p_t eps/eps_t
    ok = (eps > et) & (p > pt * eps / et)
    k = jnp.where((eps[0] > et) & (p[0] > pt), 0, jnp.argmax(ok))
    nk, ek, pk = n[k], eps[k], p[k]
    gam = jnp.log(pk / pt) / jnp.log(ek / et)
    lr = jnp.log(ek / et)
    # ln n - ln n_t = int deps/(eps + p) along the segment, on a fine s grid
    sq = jnp.linspace(0.0, 1.0, n_quad)
    eq = et * jnp.exp(sq * lr)
    integrand = eq / (eq + pt * (eq / et) ** gam) * lr               # d/ds
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(sq))])
    f = nk / (n_t * jnp.exp(cum[-1]))            # rescales crust + segment n

    def on_segment(s):
        e = et * jnp.exp(s * lr)
        return f * n_t * jnp.exp(jnp.interp(s, sq, cum)), e, pt * (e / et) ** gam

    ns, es, ps = on_segment(jnp.linspace(0.0, 1.0, n_seg + 2)[1:-1])
    # fallback only: core nodes below the segment end move onto the segment
    below = jnp.arange(n.shape[0]) < k
    sc = jnp.clip((jnp.log(n) - math.log(n_t)) / jnp.log(nk / n_t), 0.0, 1.0)
    nb, eb, pb = on_segment(sc)
    n2 = jnp.concatenate([ns, jnp.where(below, nb, n)])
    e2 = jnp.concatenate([es, jnp.where(below, eb, eps)])
    p2 = jnp.concatenate([ps, jnp.where(below, pb, p)])
    o = jnp.argsort(n2)
    return (jnp.concatenate([f * jnp.asarray(nc), n2[o]]),
            jnp.concatenate([jnp.asarray(ec), e2[o]]),
            jnp.concatenate([jnp.asarray(pc), p2[o]]))


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


def finish_table(n, eps, p, crust, n_t, n_surface, gamma_crust):
    """Low-density completion shared by the uniform-matter models, then the
    EOSTable.  ``crust="sly"``: :func:`attach_crust` (the core must start at
    :func:`join_density`).  ``crust="polytrope"``: the legacy surface continuation
    :func:`nmbackend.eos.base.extend_to_surface`.  ``n_surface=None`` skips
    both (table truncated at the core grid start)."""
    from .base import build_h_and_cs2, extend_to_surface
    if n_surface is not None:
        if crust == "sly":
            n, eps, p = attach_crust(n, eps, p, n_t=n_t)
        else:
            n, eps, p = extend_to_surface(n, eps, p, n_surface=n_surface,
                                          gamma_crust=gamma_crust)
    return build_h_and_cs2(n, eps, p)
