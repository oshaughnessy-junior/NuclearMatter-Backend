"""Isothermal cooling-curve grids for one EOS table (paper 7 population grid).

For an NSCool star EOS file (``nmbackend.nscool_io.write_tables``), build
each star with NSCool's TOV and setup (``NSCool.zone_dump``), then evaluate
the JAX isothermal curve with the crust blanket (rho_b = rhocore) for every
gap model and envelope, and interpolate log10 L_gamma^inf onto a common age
grid.  Ages count from T_b = T_max (t = 0) with no transport hand-off, so
young ages carry the early non-isothermal error (nmb-papers
``paper7_cooling_population/notes/RESULTS-G-grid-validation.md``).

Envelopes are given as physical light-element mass fractions f = Delta M / M;
each star gets PCY97's eta = g_s14^2 f (``physics/README.md``); f = 0 is iron.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from functools import partial
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .runner import GapModel, NSCool
from .physics import Star
from .isothermal import YEAR_S
from .physics.star import _l_gamma, crust_march, zone_terms
from .physics.gaps import StarTc

__all__ = ["StarCurves", "star_curves", "eos_curve_grid", "T_GRID", "T_GRID_DEFAULT",
           "TS_PCY_MIN"]

#: redshifted T_b grid (K), down to near the PCY97 formula's algebraic floor
#: (T_b,local ~ 7e3 sqrt(g_s14) K); NaN below it is trimmed.  With 500 points
#: and the defaults of star_curves, log10 L agrees with a 961-point,
#: 160-march, 20-iteration evaluation to 1e-3 dex except across n3P2 onsets
#: of strong gaps (up to 0.04 dex for SFN3P2 = 103 at 1e4-2e5 yr).
T_GRID_DEFAULT = (2.0e4, 1.0e10, 500)
#: PCY97 calibrates local T_s >= 10^4.7 K
TS_PCY_MIN = 10 ** 4.7


def T_GRID(spec=T_GRID_DEFAULT) -> np.ndarray:
    lo, hi, n = spec
    return np.geomspace(lo, hi, int(n))


class StarCurves(NamedTuple):
    """log10 L_gamma^inf [erg/s] of one star on (gap, eta, age); NaN where invalid."""
    logL: np.ndarray       # (n_gap, n_eta, n_t)
    valid: np.ndarray      # bool, same shape: age inside the curve's finite support
    t_max: np.ndarray      # (n_gap, n_eta) last finite age of the curve, yr


@partial(jax.jit, static_argnames=("n_iter",))
def _curves(star: Star, tcs: StarTc, march_tcs: StarTc, march_of_gap, etas, T, Tm,
            rho_b, n_iter=4):
    """(t_yr, L_gamma_inf) of shape (n_gap, n_eta, n_T - 1).

    Same integral as ``physics.curve.isothermal_curve(blanket=True)``, with
    the work factored: C and L_nu depend on the gap model only, L_gamma on
    eta only, and the crust march on eta and on the proton gap at the core
    boundary (``march_tcs``, one per distinct boundary Tc; ``march_of_gap``
    maps gaps to them).  The march runs on the coarse T_b grid ``Tm`` and
    ln T_in is interpolated linearly in ln T_b onto ``T``.  ``rho_b`` is
    traced, so stars that differ only in rhocore share one compilation.
    """
    z = jnp.asarray(star.zones)
    dv = star.dvol[z] + star.dvol[z + 1]
    w = star.ephi[z] ** 2 * dv

    def sums(tc):
        def at(t):
            q = zone_terms(t, star, tc)
            return jnp.sum(q["cv"] * dv), jnp.sum(q["qnu"] * w)
        return jax.vmap(at)(T)

    C, L_nu = jax.vmap(sums)(tcs)                                   # (G, nT)
    lg = jax.vmap(lambda e: jax.vmap(lambda t: _l_gamma(t, star, e)[0])(T))(etas)
    lg_m = jax.vmap(lambda e: jax.vmap(lambda t: _l_gamma(t, star, e)[0])(Tm))(etas)
    nB, nE, nM = march_tcs.tcp.shape[0], etas.shape[0], Tm.shape[0]
    Tb = jnp.broadcast_to(Tm, (nB, nE, nM)).ravel()
    L = jnp.broadcast_to(lg_m, (nB, nE, nM)).ravel()
    b = jnp.broadcast_to(jnp.arange(nB)[:, None, None], (nB, nE, nM)).ravel()

    def march(t, l, k):
        tc = jax.tree_util.tree_map(lambda a: a[k], march_tcs)
        return crust_march(t, star, l, rho_b, n_iter=n_iter, tc=tc)

    tin_m = jax.vmap(march)(Tb, L, b).reshape(nB, nE, nM)[march_of_gap]   # (G, E, nM)
    lnT, lnTm = jnp.log(T), jnp.log(Tm)
    lnTi = jax.vmap(jax.vmap(lambda y: jnp.interp(-lnT, -lnTm, jnp.log(y))))(tin_m)

    def li(y, x):                     # y on the T grid, at interior ln T x (curve.py)
        return jnp.exp(jnp.interp(-x, -lnT, jnp.log(jnp.maximum(y, 1e-300))))

    Ci = jax.vmap(lambda c, x: jax.vmap(lambda xx: li(c, xx))(x))(C, lnTi)
    Ni = jax.vmap(lambda c, x: jax.vmap(lambda xx: li(c, xx))(x))(L_nu, lnTi)
    f = Ci / (Ni + lg[None]) * jnp.exp(lnTi)
    t = jnp.cumsum(0.5 * (f[..., 1:] + f[..., :-1]) * -jnp.diff(lnTi, axis=-1), axis=-1)
    return t / YEAR_S, jnp.broadcast_to(lg[None, :, 1:], t.shape)


def _on_ages(t, L, t_grid):
    """log10 L on t_grid (log-log), restricted to the leading finite, increasing run."""
    ok = np.isfinite(t) & np.isfinite(L) & (L > 0) & (t > 0)
    bad = np.flatnonzero(~ok | np.r_[False, np.diff(t) <= 0])
    # leading run only: drop from the first bad point after the first good one
    first = int(np.argmax(ok))
    bad = bad[bad > first]
    end = bad[0] if bad.size else t.size
    tt, LL = t[first:end], L[first:end]
    out = np.full(t_grid.size, np.nan)
    if tt.size < 2:
        return out, np.zeros(t_grid.size, bool), np.nan
    v = (t_grid >= tt[0]) & (t_grid <= tt[-1])
    out[v] = np.interp(np.log(t_grid[v]), np.log(tt), np.log10(LL))
    return out, v, float(tt[-1])


def star_curves(star: Star, gaps: Sequence[GapModel], eta_pcy, t_grid,
                T_grid=None, n_march=80, n_iter=4) -> StarCurves:
    """Curves of one star for each gap model and each PCY97 eta (JAX, jitted).

    All gap models must have ``ca_n3p2`` = 0.5.  ``eta_pcy``: 1-D array (0 = iron).
    The crust march runs on ``n_march`` log-spaced T_b over the range of
    ``T_grid`` with ``n_iter`` Picard iterations per zone.  Against 320
    marches and 30 iterations, the defaults change log10 L by < 1e-3 dex
    (APR 1.4 and 2.0, all eight paper-7 gap sets); 4 and 10 iterations agree
    exactly.
    """
    if {g.ca_n3p2 for g in gaps} != {0.5}:
        raise ValueError("star_curves uses ca_n3p2 = 0.5 for every gap model")
    T = np.sort(T_GRID() if T_grid is None else np.asarray(T_grid, float))[::-1]
    Tm = np.geomspace(T[0], T[-1], int(n_march))
    tcs = [star.tc(g) for g in gaps]
    # the march sees the gaps only through tcp at the core boundary zone icore
    key = [float(tc.tcp[star.icore]) for tc in tcs]
    uniq = sorted(set(key))
    pick = [key.index(u) for u in uniq]
    stack = lambda L: StarTc(*(jnp.stack([getattr(tc, f) for tc in L]) for f in StarTc._fields))
    rho_b = float(star.rhocore)
    star = dataclasses.replace(star, rhocore=0.0)   # static field: avoid a recompile per rhocore
    t, L = _curves(star, stack(tcs), stack([tcs[i] for i in pick]),
                   jnp.asarray([uniq.index(k) for k in key]),
                   jnp.asarray(eta_pcy, float), jnp.asarray(T), jnp.asarray(Tm),
                   jnp.asarray(rho_b), int(n_iter))
    t, L = np.asarray(t), np.asarray(L)
    t_grid = np.asarray(t_grid, float)
    shape = t.shape[:2]
    logL = np.full(shape + (t_grid.size,), np.nan)
    valid = np.zeros(logL.shape, bool)
    tmax = np.full(shape, np.nan)
    for i in range(shape[0]):
        for j in range(shape[1]):
            logL[i, j], valid[i, j], tmax[i, j] = _on_ages(t[i, j], L[i, j], t_grid)
    return StarCurves(logL, valid, tmax)


def eos_curve_grid(eos_table, masses, gaps: Sequence[GapModel], f_env, t_grid, *,
                   rhocore=1.6e14, T_grid=None, nscool: NSCool | None = None,
                   nc_seed=None, timeout=120.0) -> dict:
    """Curve grid of one EOS: log10 L_gamma^inf on (mass, gap, envelope, age).

    ``eos_table``: NSCool star EOS file (None: the bundled APR).  ``f_env``: physical light-element
    mass fractions (0 = iron); per star eta_PCY = g_s14^2 f.  Stars NSCool
    cannot build (e.g. past its own maximum mass) get NaN rows and
    ``star_ok`` False.  ``nc_seed`` (n_M,), central baryon densities in
    fm^-3 (e.g. from nmbackend's TOV), seeds NSCool's target-mass search;
    its default start (0.3 fm^-3) can step past the table and fail.  Returns dict with logL (n_M, n_gap, n_eta, n_t),
    valid, t_max, gs14, radius_cm (NSCool zone imax at 1e10 g/cm^3, as _l_gamma),
    exp_phi_s, eta_pcy (n_M, n_eta), n_durca_zones, star_ok and
    logL_pcy_floor (n_M,): log10 L_gamma^inf at local T_s = 10^4.7 K, below
    which PCY97 is extrapolated.  T_eff^inf = (L / (4 pi sigma radius^2))^(1/4)
    / exp_phi_s^(1/2) reproduces NSCool's convention.
    """
    ns = nscool or NSCool()
    eos_table = str(eos_table or ns.default_eos)
    masses = np.asarray(masses, float)
    f_env = np.asarray(f_env, float)
    t_grid = np.asarray(t_grid, float)
    nM, nG, nE, nT = masses.size, len(gaps), f_env.size, t_grid.size
    out = dict(logL=np.full((nM, nG, nE, nT), np.nan),
               valid=np.zeros((nM, nG, nE, nT), bool),
               t_max=np.full((nM, nG, nE), np.nan), gs14=np.full(nM, np.nan),
               radius_cm=np.full(nM, np.nan), exp_phi_s=np.full(nM, np.nan),
               eta_pcy=np.full((nM, nE), np.nan), n_durca_zones=np.zeros(nM, int),
               star_ok=np.zeros(nM, bool), logL_pcy_floor=np.full(nM, np.nan),
               error=[""] * nM)
    seeds = np.full(nM, 0.3) if nc_seed is None else np.asarray(nc_seed, float)
    work = tempfile.mkdtemp(prefix="grid_", dir=ns.workroot)
    for k, M in enumerate(masses):
        try:
            prof = os.path.join(work, "prof.dat")
            ns.make_profile(eos_table, float(M), prof, rhoc_start=float(seeds[k]),
                            timeout=timeout)
            d = ns.zone_dump([1e8], profile=prof, eos_table=eos_table,
                             rhocore=rhocore, rho_b=rhocore, timeout=timeout)
        except Exception as e:          # NSCoolError, TOV past M_max, PAUSE
            out["error"][k] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        s = Star.from_dump(d)
        eta = d.gs14 ** 2 * f_env
        c = star_curves(s, gaps, eta, t_grid, T_grid)
        out["logL"][k], out["valid"][k], out["t_max"][k] = c
        im = d.imax
        out["gs14"][k], out["radius_cm"][k] = d.gs14, float(d.star["rad"][im])
        out["exp_phi_s"][k] = float(d.star["ephi"][im - 1])   # as _l_gamma for Teff_inf
        out["eta_pcy"][k] = eta
        out["n_durca_zones"][k] = int(np.sum(d.star["idurca_np"][d.zones] > 0))
        e_s = float(d.star["ephi"][im])           # L_gamma^inf = L_local e_s^2
        out["logL_pcy_floor"][k] = np.log10(4 * np.pi * out["radius_cm"][k] ** 2 * 5.67e-5
                                            * TS_PCY_MIN ** 4 * e_s ** 2)
        out["star_ok"][k] = True
    shutil.rmtree(work, ignore_errors=True)
    return out
