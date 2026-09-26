"""Tabulated SLy crust joined to a uniform-matter core.

Crust: the SLy unified EOS (BPS + Haensel-Pichon outer crust, Douchin &
Haensel 2001 inner crust, astro-ph/0111092), ``data/crust_sly_dh2001.dat``.
Below ``n_t`` the table is used as is.  This is the fixed crust of the paper-1
crust+cEFT ensemble (n_t = 0.055 fm^-3 = 0.34 n_sat).

Join (the P(rho) matching of Fortin et al. 2016, arXiv:1604.01944, Sec. II.B):

* Core nodes (n, eps, p) at and above the first node n_j >= ``bridge`` * n_t
  whose pressure exceeds p_t (eps_j/eps_t)^GAMMA_MIN are kept unchanged.
* Between the crust point (eps_t, p_t) and (eps_j, p_j), p is a power law in
  eps, so p is continuous and strictly increasing in eps.
* n follows the first law, d ln n = d eps/(eps + p), along the crust and the
  power-law segment.  It then misses the core n_j by a constant factor f
  (|f - 1| is 1e-3 to 1e-2).  We rescale the crust and segment n by f
  (Fortin et al. Eq. 11, applied on the crust side).

The TOV equations see only p(eps), which is the core's own above n_j; the
core baryon density, and hence the composition and dUrca onset density, are
untouched.  Vary ``n_t`` to estimate the crust-matching systematic.
"""
from __future__ import annotations

import math
import os

import numpy as np
import jax.numpy as jnp

N_T_DEFAULT = 0.055       # fm^-3, 0.34 n_sat (paper 1, GKKS arXiv:2512.18044)
N_FLOOR = 1e-10           # fm^-3, lowest crust node (surface enthalpy ~ 1e-5)
BRIDGE = 1.25             # power-law segment ends at >= BRIDGE * n_t
GAMMA_MIN = 1.1           # minimum power-law index on that segment
N_CRUST_PTS = 150

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


def attach_crust(n, eps, p, n_t=N_T_DEFAULT, bridge=BRIDGE, gamma_min=GAMMA_MIN,
                 n_floor=N_FLOOR, n_crust_pts=N_CRUST_PTS, n_quad=257):
    """Join the SLy crust below ``n_t`` to a core (n, eps, p) given on nodes
    n > n_t (fm^-3, MeV/fm^3).  JAX-traceable in the core arrays; the node
    count is static.  Returns the combined (n, eps, p)."""
    n, eps, p = jnp.asarray(n), jnp.asarray(eps), jnp.asarray(p)
    nc, ec, pc = crust_nodes(n_t, n_floor, n_crust_pts)
    et, pt = ec[-1], pc[-1]
    # first core node past the bridge start that a power law p(eps) reaches
    ok = (n >= bridge * n_t) & (eps > et) & (p >= pt * (eps / et) ** gamma_min)
    j = jnp.argmax(ok)
    ej, pj, nj = eps[j], p[j], n[j]
    gam = jnp.log(pj / pt) / jnp.log(ej / et)
    # segment nodes: the core slots i < j, placed at eps_t (eps_j/eps_t)^s
    s = jnp.clip((jnp.log(n) - math.log(n_t)) / (jnp.log(nj) - math.log(n_t)),
                 0.0, 1.0)
    e_br = et * (ej / et) ** s
    p_br = pt * (e_br / et) ** gam
    # ln n(eps) - ln n_t = int_{eps_t}^{eps} deps/(eps + p) on a fine s grid
    sq = jnp.linspace(0.0, 1.0, n_quad)
    eq = et * (ej / et) ** sq
    integrand = eq / (eq + pt * (eq / et) ** gam) * jnp.log(ej / et)   # d/ds
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(sq))])
    f = nj / (n_t * jnp.exp(cum[-1]))            # rescales crust + segment n
    n_br = f * n_t * jnp.exp(jnp.interp(s, sq, cum))
    below = jnp.arange(n.shape[0]) < j
    return (jnp.concatenate([f * jnp.asarray(nc), jnp.where(below, n_br, n)]),
            jnp.concatenate([jnp.asarray(ec), jnp.where(below, e_br, eps)]),
            jnp.concatenate([jnp.asarray(pc), jnp.where(below, p_br, p)]))


def core_grid(n_grid, n_t=N_T_DEFAULT):
    """The nodes of a static core grid that lie above ``n_t``."""
    g = np.asarray(n_grid)
    return jnp.asarray(g[g > n_t * (1.0 + 1e-9)])


CRUSTS = ("sly", "polytrope")


def check_crust(crust):
    if crust not in CRUSTS:
        raise ValueError(f"crust={crust!r}; use one of {CRUSTS}")
    return crust


def finish_table(n, eps, p, crust, n_t, n_surface, gamma_crust):
    """Low-density completion shared by the uniform-matter models, then the
    EOSTable.  ``crust="sly"``: :func:`attach_crust` (core nodes must lie
    above n_t).  ``crust="polytrope"``: the legacy surface continuation
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
