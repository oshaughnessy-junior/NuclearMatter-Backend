"""The tidal term is consistent with the eps(h) of the structure solve.

The Love-number equation needs (eps + p)/cs2 = (eps + p) deps/dp along the
star.  The structure takes eps(h) from a cubic Hermite of p(h); until
2026-10-14 the tidal term took cs2 from finite-difference node values
interpolated linearly in h, which is not the derivative of that eps(h).  Across
a steep cell (p x2.7 while eps rises 11%, as at Q2C's crust-core joins) the
stiff node value spread into the soft cell, the tidal source missed part of the
cell's Delta eps, and Lambda(1.4) came out a few % low.  Now the tidal term is
the analytic deps/dh of the same Hermite cell, and the Hermite's cell widths in
h come from the log-linear eps(p) between the nodes rather than the trapezoid
(docs/devlog/2026-10-14-tidal-cs2.md).

References (both from the independent radius-domain solver, SciPy RK45 with
the (H, beta) tidal system, cs2 the slope of its own log-linear eps(p)):
  * the table's Hermite eps(h), p(h) resampled densely -- the EOS the solver
    integrates, without the analytic derivative;
  * the raw table nodes -- the log-linear (piecewise-polytrope) EOS that RIFT
    and the Q2C oracle use.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from nmbackend.constants import M_BARYON, MEVFM3_TO_GEOM
from nmbackend.eos.base import (EOSTable, build_h_and_cs2, hermite_eos_at_h,
                                hermite_eos_tidal_at_h)
from nmbackend.eos.crust import sly_crust
from nmbackend.reference.scipy_tov import solve_reference
from nmbackend.tov.solver import solve_structure

_FRAC, _RATIO, _NJ, _GAMMA = 0.9, 2.7, 0.08, 2.5
_HC = None            # the M = 1.4 star of the steep table (_hc14; by root, so it
                      # does not depend on GMSUN_KM)


def _steep_table(scale=1.0, n_crust=80, n_core=200):
    """SLy crust up to eps_x = 0.9 eps_j, then a polytropic core
    p = scale K n^2.5 from n_j = 0.08 fm^-3 whose first node has p = 2.7 p_x:
    one cell where p rises x2.7 while eps rises 11% (a Q2C-style 'gap' join)."""
    n0, e0, p0 = sly_crust()
    ln0, le0, lp0 = np.log(n0), np.log(e0), np.log(p0)
    eps_j = _NJ * M_BARYON
    for _ in range(50):
        eps_x = _FRAC * eps_j
        p_x = math.exp(np.interp(math.log(eps_x), le0, lp0))
        eps_j = _NJ * M_BARYON + _RATIO * p_x / (_GAMMA - 1.0)
    K = _RATIO * p_x / _NJ ** _GAMMA
    n_x = math.exp(np.interp(math.log(eps_x), le0, ln0))
    nc = np.geomspace(3e-10, n_x, n_crust)
    ec = np.exp(np.interp(np.log(nc), ln0, le0))
    pc = np.exp(np.interp(np.log(nc), ln0, lp0))
    ec[-1], pc[-1] = eps_x, p_x
    nk = np.geomspace(_NJ, 1.6, n_core)
    pk = scale * K * jnp.asarray(nk) ** _GAMMA
    ek = nk * M_BARYON + pk / (_GAMMA - 1.0)
    return build_h_and_cs2(jnp.concatenate([jnp.asarray(nc), jnp.asarray(nk)]),
                           jnp.concatenate([jnp.asarray(ec), ek]),
                           jnp.concatenate([jnp.asarray(pc), pk]))


def _dense(tab, K):
    """The table's Hermite (eps, p)(h), sampled K times per cell."""
    h, pg, eg, c = (np.asarray(a, float) for a in (tab.h, tab.p, tab.eps, tab.cs2))
    t = np.arange(K) / K
    hq = np.append((h[:-1, None] + t[None, :] * np.diff(h)[:, None]).ravel(), h[-1])
    e, p, _ = hermite_eos_at_h(h, pg, eg, c, hq, xp=np)
    one = np.ones_like(hq)
    return EOSTable(n=one, eps=e, p=p, h=hq, cs2=one, eps_mevfm3=e / MEVFM3_TO_GEOM,
                    p_mevfm3=p / MEVFM3_TO_GEOM)


def _obs(scale, hc, rtol, atol):
    pt = solve_structure(_steep_table(scale), hc, rtol=rtol, atol=atol)
    return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])


_val = jax.jit(_obs)


def _hc14():
    """h_c of the M = 1.4 star of the steep table (structure-only root)."""
    global _HC
    if _HC is None:
        from scipy.optimize import brentq
        m = jax.jit(lambda h: solve_structure(_steep_table(), h, tidal=False,
                                              rtol=1e-10, atol=1e-12).M_msun)
        _HC = brentq(lambda h: float(m(h)) - 1.4, 0.19, 0.24, xtol=1e-14)
    return _HC


def test_tidal_source_integrates_to_the_cell_delta_eps():
    # Per cell, int deps/dh dh == Delta eps (Gauss-Legendre is exact for the
    # quadratic), and eps, p equal those of the structure interpolant bit for bit.
    tab = _steep_table()
    h, pg, eg, c = (np.asarray(a) for a in (tab.h, tab.p, tab.eps, tab.cs2))
    x, w = np.polynomial.legendre.leggauss(4)
    s, w = 0.5 * (x + 1.0), 0.5 * w
    dh = np.diff(h)
    hq = (h[:-1, None] + s[None, :] * dh[:, None]).ravel()
    e1, p1, d = hermite_eos_tidal_at_h(h, pg, eg, hq, xp=np)
    e0, p0, _ = hermite_eos_at_h(h, pg, eg, c, hq, xp=np)
    assert np.array_equal(e1, e0) and np.array_equal(p1, p0)
    integ = (w[None, :] * d.reshape(-1, 4)).sum(1) * dh
    np.testing.assert_allclose(integ, np.diff(eg), rtol=1e-9, atol=0.0)
    assert np.all(d > 0.0)                        # monotone eps(h) on this table


def test_lambda_matches_references_on_steep_joint():
    # Before 2026-10-14: Lambda 589.9 at M = 1.4, -4.7% (a phantom -5.1 MeV/fm^3
    # of Delta eps at the joint), and R +22 m (trapezoid h).  Measured now:
    # vs the dense-Hermite reference 2e-8, 7e-8, 9e-8 in (M, R, Lambda); vs the
    # raw-node log-linear reference 6e-7, 1e-7, 7e-7.
    hc = _hc14()
    v = np.asarray(_val(1.0, hc, 1e-10, 1e-12))
    assert abs(v[0] - 1.4) < 1e-8
    tab = _steep_table()
    # the raw-node reference interpolates log p linearly in h: give it the same
    # central pressure (its own h_c), so both integrate the same star
    p_c = float(tab.at_h(hc)[1])
    h, lp = np.asarray(tab.h), np.log(np.asarray(tab.p))
    for table, hr, tol in ((_dense(tab, 16), hc, 1e-6),
                           (tab, np.interp(math.log(p_c), lp, h), 5e-6)):
        ref = solve_reference(table, hr, rtol=1e-10, atol=1e-40, consistent_cs2=True)
        np.testing.assert_allclose(v, [ref.M_msun, ref.R_km, ref.Lambda],
                                   rtol=tol, atol=0.0)


def test_lambda_tangent_keeps_the_node_boundary_terms():
    # The tidal term jumps at the nodes; nodes above the joint move with the
    # core scale.  Without the node-anchored y picture the forward-mode
    # dLambda/dscale is off by a factor 2.5 here; measured now 1.5e-7 at the
    # default rtol (dM, dR: 1.4e-8, 4.4e-9).
    hc = _hc14()
    jvp = jax.jit(lambda s: jax.jvp(lambda x: _obs(x, hc, 1e-7, 1e-9), (s,), (1.0,))[1])
    ad = np.asarray(jvp(1.0))
    d = 1e-4
    fd = (np.asarray(_val(1.0 + d, hc, 1e-11, 1e-13))
          - np.asarray(_val(1.0 - d, hc, 1e-11, 1e-13))) / (2 * d)
    np.testing.assert_allclose(ad[:2], fd[:2], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(ad[2], fd[2], rtol=5e-6, atol=0.0)


def test_bridge_density_jump_gets_the_junction():
    """Every draw_bridge_eos table has a density jump at fixed p at the
    low-table/bridge join (n_L = 0.32 fm^-3: p 17.7128 at both nodes, eps
    310.75 -> 312.73 MeV/fm^3, cell width 1.7e-16 in h).  There the cubic's
    deps/dh is +-1e18 noise; it used to make tidal solves at rtol 1e-10 hit
    max_steps.  The cell now carries no tidal term and its Delta eps goes into
    the cell below (the junction bubble, within the top ~3% of that cell).
    Measured on draw 1: Lambda 224.0959 vs the reference with the exact junction
    224.0955 (1.8e-6; 1.8e-6..2.3e-6 on 4 stars of 2 draws, 1.2e-5..3.2e-5 with
    the first, centred bubble) and 223.5689 without it (main dropped the jump
    too: 224.028 at rtol 1e-10, with its phantom term)."""
    from nmbackend.priors.bridge import draw_bridge_eos
    from nmbackend.tov import build_family
    from nmbackend.tov.family import default_hc_grid
    t = draw_bridge_eos(np.random.default_rng(1), corr_frac=0.3)["table"]
    hc = 0.29
    v = np.asarray(_val_table(t, hc))
    p_c = float(t.at_h(hc)[1])
    hr = np.interp(math.log(p_c), np.log(np.asarray(t.p)), np.asarray(t.h))
    ref = solve_reference(t, hr, rtol=1e-10, atol=1e-40, consistent_cs2=True)
    np.testing.assert_allclose(v, [ref.M_msun, ref.R_km, ref.Lambda], rtol=2e-5,
                               atol=0.0)
    # and the batched family at the same tolerance (one bad star would raise)
    fam = build_family(t, default_hc_grid(t, 8), rtol=1e-10, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(fam.Lambda)))


_val_table = jax.jit(lambda t, hc: (lambda s: jnp.stack([s.M_msun, s.R_km, s.Lambda]))(
    solve_structure(t, hc, rtol=1e-10, atol=1e-12)))
