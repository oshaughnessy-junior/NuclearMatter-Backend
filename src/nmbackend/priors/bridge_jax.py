"""Differentiable version of the constrained Brownian-bridge prior (bridge.py).

:func:`bridge.draw_bridge_eos` draws its random numbers from a NumPy stream
while it builds the EOS.  Here the random numbers are explicit arguments, so
one draw is a deterministic function

    table = bridge_table(nep, corr_frac, ln_X, z)

of the nuclear empirical parameters ``nep`` (through the metamodel anchor at
n_L), the sound-speed correlation length, the pQCD scale, and the standard-
normal variates ``z``.  Every step is written in JAX, so the map can be
differentiated with respect to all of these and composed with the TOV solver
(for example, a Fisher matrix with the variates as nuisance parameters whose
prior is a unit normal).

Differences from the NumPy sampler (same prior law, different realization):

* variates: each node of the refinement tree owns three normals
  (heap index k = 1 .. 2^levels - 1), mapped to uniforms by the normal CDF.
  ``u_mu`` places mu by the inverse CDF of the KK marginal; ``u_n`` places n
  by ONE inverse CDF over the whole tent-shaped marginal (the NumPy sampler
  uses one uniform to pick the rising or falling side and another inside it);
  ``u_p`` places p uniformly in the slice.  The tent map is continuous at the
  apex, so the draw is smooth in its arguments away from the checks below.
* the tree is built level by level.  Where the NumPy sampler skips a node (an
  inconsistent interval, ln(mu_b/mu_a) < min_dlnmu, or a node outside its
  parent interval), the node here collapses onto its left endpoint and its
  descendants are skipped, which gives the same piecewise mu(n).
* the table is built with :func:`eos.base.build_h_and_cs2` from the raw nodes
  (no :func:`eos.base.refine_table`), so its length is set by the bridge grid.

The checks above, the running maximum on mu, and the truncation at n_term
make the map piecewise smooth.  Derivatives are exact inside a piece; the
indices that select a piece are fixed by the concrete values at the point of
evaluation (:func:`plan`), so tracing never branches on a traced value.

Units as bridge.py: mu in MeV, n in fm^-3, p and eps in MeV/fm^3.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from .bridge import MU_H, N_SAT, PQCD_TABLE

_XS = np.array(sorted(PQCD_TABLE))
_C_N = np.polyfit(np.log(_XS), [PQCD_TABLE[x][0] for x in _XS], 2)
_C_P = np.polyfit(np.log(_XS), [PQCD_TABLE[x][1] for x in _XS], 2)
_U_EPS = 1e-12


def pqcd_anchor(ln_X):
    """(mu_H, n_H, p_H) at mu = 2.6 GeV as bridge.pqcd_anchor, for ln X in [0, ln 4]."""
    x = jnp.clip(ln_X, 0.0, math.log(4.0))
    return (jnp.asarray(MU_H, x.dtype), jnp.polyval(jnp.asarray(_C_N), x),
            jnp.polyval(jnp.asarray(_C_P), x))


# ---------------------------------------------------------------- KK volume ------
def _kk(bL, bH):
    muL, nL, pL = bL
    muH, nH, pH = bH
    dp = pH - pL
    den = muL * nH - muH * nL
    num = muL * muH * (muH * nH - muL * nL - 2.0 * dp)
    ok = (den > 0.0) & (num > 0.0)
    ratio = jnp.where(ok, num / jnp.where(den > 0.0, den, 1.0), muL ** 2)
    mu_c = jnp.where(ok, jnp.sqrt(jnp.clip(ratio, muL ** 2, muH ** 2)),
                     jnp.where(den <= 0.0, muH, muL))

    def n_min(mu):
        a = nL * mu / muL
        d = (mu ** 2 - muL ** 2) * muH
        b = (mu ** 3 * nH - mu * muH * (muH * nH - 2.0 * dp)) / jnp.where(d != 0.0, d, 1.0)
        return jnp.where(mu <= mu_c, a, b)

    def n_max(mu):
        a = nH * mu / muH
        d = (mu ** 2 - muH ** 2) * muL
        b = (mu ** 3 * nL - muL * mu * (muL * nL + 2.0 * dp)) / jnp.where(d != 0.0, d, 1.0)
        return jnp.where(mu >= mu_c, a, b)

    def p_min(mu):
        return pL + (mu ** 2 - muL ** 2) / (2.0 * mu) * n_min(mu)

    n_c0 = n_max(muL) / muL

    def p_max(mu, n):
        return jnp.where(n <= n_c0 * mu,
                         pL + (mu ** 2 - muL ** 2) / (2.0 * mu) * n,
                         pH - (muH ** 2 - mu ** 2) / (2.0 * mu) * n)

    return n_min, n_max, p_min, p_max, mu_c, n_c0


def _consistent(a, b, rtol=1e-9):
    muL, nL, pL = a
    muH, nH, pH = b
    dp = pH - pL
    span = muH ** 2 - muL ** 2
    lo = nL * span / (2.0 * muL)
    hi = nH * span / (2.0 * muH)
    tol = rtol * jnp.maximum(jnp.abs(hi), 1.0)
    return (muH > muL) & (nH >= nL) & (lo - tol <= dp) & (dp <= hi + tol)


def _sample(a, b, u, n_cdf=256):
    """One point in the KK volume between triplets a and b from uniforms u[3]."""
    muL, muH = a[0], b[0]
    n_min, n_max, p_min, p_max, mu_c, n_c0 = _kk(a, b)
    m2 = jnp.linspace(muL ** 2, muH ** 2, n_cdf)
    mc2, mL2, mH2 = mu_c ** 2, muL ** 2, muH ** 2

    def sdiv(x, y):
        good = y > 0.0
        return jnp.where(good, x / jnp.where(good, y, 1.0), 0.0)

    dens = jnp.where(m2 <= mc2,
                     sdiv((m2 - mL2) * (mH2 - mc2), (mc2 - mL2) * (mH2 - m2)),
                     sdiv((mH2 - m2) * (mc2 - mL2), (mH2 - mc2) * (m2 - mL2)))
    dens = jnp.where(jnp.isfinite(dens), jnp.clip(dens, 0.0, None), 0.0)
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (dens[1:] + dens[:-1])
                                                    * jnp.diff(m2))])
    tot = cdf[-1]
    good = jnp.isfinite(tot) & (tot > 0.0)
    mu2 = jnp.where(good, _interp(u[0] * jnp.where(good, tot, 1.0), cdf, m2),
                    0.5 * (mL2 + mH2))
    mu = jnp.sqrt(mu2)

    # n: one inverse CDF over the tent h(n) = p_max(mu, n) - p_min(mu)
    n_lo, n_hi = n_min(mu), n_max(mu)
    n_apex = jnp.minimum(jnp.maximum(n_c0 * mu, n_lo), n_hi)
    pmin = p_min(mu)
    h_apex = jnp.maximum(p_max(mu, n_apex) - pmin, 0.0)
    h_lo = jnp.maximum(p_max(mu, n_lo) - pmin, 0.0)
    h_hi = jnp.maximum(p_max(mu, n_hi) - pmin, 0.0)
    A1 = 0.5 * (h_lo + h_apex) * jnp.maximum(n_apex - n_lo, 0.0)
    A2 = 0.5 * (h_apex + h_hi) * jnp.maximum(n_hi - n_apex, 0.0)
    At = A1 + A2
    w = jnp.where(At > 0.0, A1 / jnp.where(At > 0.0, At, 1.0), 0.5)

    def trap(x0, x1, h0, h1, s):
        # inverse CDF of a linear density from h0 at x0 to h1 at x1, s in [0, 1]
        dh = h1 - h0
        flat = jnp.abs(dh) <= 1e-14 * jnp.maximum(jnp.abs(h1) + jnp.abs(h0), 1e-300)
        root = jnp.sqrt(jnp.maximum(h0 ** 2 + s * (h1 ** 2 - h0 ** 2), 0.0))
        frac = jnp.where(flat, s, (root - h0) / jnp.where(flat, 1.0, dh))
        return x0 + (x1 - x0) * frac

    s1 = jnp.clip(u[1] / jnp.where(w > 0.0, w, 1.0), 0.0, 1.0)
    s2 = jnp.clip((u[1] - w) / jnp.where(w < 1.0, 1.0 - w, 1.0), 0.0, 1.0)
    n = jnp.where(u[1] < w, trap(n_lo, n_apex, h_lo, h_apex, s1),
                  trap(n_apex, n_hi, h_apex, h_hi, s2))
    n = jnp.where(At > 0.0, n, 0.5 * (n_lo + n_hi))
    p = pmin + u[2] * jnp.maximum(p_max(mu, n) - pmin, 0.0)
    return jnp.stack([mu, n, p])


def _interp(x, xp, fp):
    """Linear interpolation that stays finite (and has finite derivatives)
    when xp has repeated values."""
    i = jnp.clip(jnp.searchsorted(jax.lax.stop_gradient(xp), jax.lax.stop_gradient(x),
                                  side="right"), 1, xp.shape[0] - 1)
    x0, x1 = xp[i - 1], xp[i]
    dx = x1 - x0
    t = jnp.where(dx > 0.0, (x - x0) / jnp.where(dx > 0.0, dx, 1.0), 0.0)
    return fp[i - 1] + jnp.clip(t, 0.0, 1.0) * (fp[i] - fp[i - 1])


# ----------------------------------------------------------- refinement tree -----
def refine_tree(bL, bH, U, levels=9, min_dlnmu=1e-4):
    """Level-by-level fractal refinement with fixed uniforms ``U`` of shape
    (2^levels, 3) (row 0 unused; row k is heap node k).  Returns the points
    (2^levels + 1, 3) in mu order, with skipped nodes collapsed onto their left
    endpoint, and the per-node validity mask (heap order)."""
    pts = jnp.stack([jnp.stack(bL), jnp.stack(bH)])
    live = jnp.ones(1, bool)
    valid = [jnp.zeros(1, bool)]
    for lev in range(levels):
        a, b = pts[:-1], pts[1:]
        k0 = 2 ** lev
        Uk = U[k0:2 * k0]
        ok = live & (jnp.log(b[:, 0] / a[:, 0]) >= min_dlnmu)
        ok = ok & jax.vmap(lambda x, y: _consistent(tuple(x), tuple(y)))(a, b)
        # a skipped interval is sampled on the root interval instead, and the
        # result discarded: sampling a degenerate interval gives NaN, which
        # would reach the derivatives through jnp.where
        sa = jnp.where(ok[:, None], a, pts[0])
        sb = jnp.where(ok[:, None], b, pts[-1])
        m = jax.vmap(lambda x, y, u: _sample(tuple(x), tuple(y), u))(sa, sb, Uk)
        inside = ((a[:, 0] < m[:, 0]) & (m[:, 0] < b[:, 0]) & (a[:, 1] <= m[:, 1])
                  & (m[:, 1] <= b[:, 1]) & (a[:, 2] <= m[:, 2]) & (m[:, 2] <= b[:, 2]))
        ok = ok & inside & jnp.all(jnp.isfinite(m), axis=1)
        m = jnp.where(ok[:, None], m, a)
        valid.append(ok)
        new = jnp.empty((2 * a.shape[0] + 1, 3), pts.dtype)
        new = new.at[0::2].set(pts).at[1::2].set(m)
        pts = new
        live = jnp.repeat(ok, 2)
    return pts, jnp.concatenate(valid)


# ------------------------------------------------------------- diffusion ---------
def _thomas(a, b, c, d):
    def fwd(carry, x):
        cp, dp = carry
        ai, bi, ci, di = x
        m = bi - ai * cp
        cn = ci / m
        dn = (di - ai * dp) / m
        return (cn, dn), (cn, dn)

    (_, _), (cp, dp) = jax.lax.scan(fwd, (0.0 * d[0], 0.0 * d[0]), (a, b, c, d))

    def bwd(xn, x):
        cpi, dpi = x
        xi = dpi - cpi * xn
        return xi, xi

    _, xs = jax.lax.scan(bwd, dp[-1], (cp[:-1], dp[:-1]), reverse=True)
    return jnp.concatenate([xs, dp[-1:]])


def diffuse_mu_of_n(x, mu, corr_frac, n_steps=40):
    """Backward-Euler diffusion of ln mu in x = ln n, Dirichlet ends (bridge.py)."""
    dx = x[1] - x[0]
    u = jnp.log(mu)
    N = x.shape[0]
    D = 0.25 * corr_frac ** 2
    r = (1.0 / n_steps) * D / dx ** 2
    a = jnp.full(N, -r).at[0].set(0.0).at[-1].set(0.0)
    c = jnp.full(N, -r).at[0].set(0.0).at[-1].set(0.0)
    b = jnp.full(N, 1.0 + 2.0 * r).at[0].set(1.0).at[-1].set(1.0)

    def step(uu, _):
        return _thomas(a, b, c, uu), None

    u, _ = jax.lax.scan(step, u, None, length=n_steps)
    return jnp.exp(u)


# ------------------------------------------------------------- one draw ----------
class Plan(NamedTuple):
    """Static choices fixed by the concrete values at the evaluation point."""
    n_keep: int          # low-table rows strictly below n_L
    n_sel: int           # bridge-grid rows at or below n_term


class Config(NamedTuple):
    levels: int = 9
    n_L_fac: float = 2.0
    n_term_fac: float = 12.0
    n_grid_pts: int = 600
    n_steps: int = 40
    min_dlnmu: float = 1e-4


def unpack(theta, nep_names, levels):
    """theta = [nep..., corr_frac, ln_X, z (2^levels - 1, 3) flattened]."""
    k = len(nep_names)
    nep = {nm: theta[i] for i, nm in enumerate(nep_names)}
    z = theta[k + 2:].reshape(2 ** levels - 1, 3)
    return nep, theta[k], theta[k + 1], z


def bridge_arrays(model, theta, nep_names, plan: Plan, cfg: Config = Config()):
    """(n, eps, p) of one draw, MeV/fm^3, as JAX arrays of static length."""
    nep, corr_frac, ln_X, z = unpack(theta, nep_names, cfg.levels)
    n_L = cfg.n_L_fac * N_SAT
    t = model.eos_table(nep)
    n_lo, eps_lo, p_lo = t.n, t.eps_mevfm3, t.p_mevfm3
    eps_L = jnp.interp(n_L, n_lo, eps_lo)
    p_L = jnp.interp(n_L, n_lo, p_lo)
    mu_L = (eps_L + p_L) / n_L
    bL = (mu_L, jnp.asarray(n_L), p_L)
    bH = pqcd_anchor(ln_X)

    U = jnp.concatenate([jnp.full((1, 3), 0.5),
                         jnp.clip(ndtr(z), _U_EPS, 1.0 - _U_EPS)])
    pts, _ = refine_tree(bL, bH, U, cfg.levels, cfg.min_dlnmu)
    x = jnp.linspace(math.log(n_L), jnp.log(bH[1]), cfg.n_grid_pts)
    mu_grid = jnp.exp(_interp(x, jnp.log(pts[:, 1]), jnp.log(pts[:, 0])))
    mu_s = jax.lax.cummax(diffuse_mu_of_n(x, mu_grid, corr_frac, cfg.n_steps))
    grid = jnp.exp(x)
    eps_b = eps_L + jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (mu_s[1:] + mu_s[:-1]) * jnp.diff(grid))])
    p_b = mu_s * grid - eps_b
    k, s = plan.n_keep, plan.n_sel
    n = jnp.concatenate([n_lo[:k], grid[:s]])
    eps = jnp.concatenate([eps_lo[:k], eps_b[:s]])
    p = jnp.concatenate([p_lo[:k], p_b[:s]])
    return n, eps, p, dict(mu_L=mu_L, p_L=p_L, p_end_mismatch=(p_b[-1] - bH[2]) / bH[2],
                           grid=grid, mu=mu_s, pts=pts)


def make_plan(model, theta, nep_names, cfg: Config = Config()):
    """Concrete pass: the static row counts at this evaluation point."""
    nep, _, ln_X, _ = unpack(np.asarray(theta), nep_names, cfg.levels)
    n_L = cfg.n_L_fac * N_SAT
    t = model.eos_table({k: float(v) for k, v in nep.items()})
    n_lo = np.asarray(t.n)
    n_keep = int(np.sum(n_lo < n_L * (1.0 - 1e-12)))
    # one length for every X: the grid rows up to n_term for the largest n_H
    # (fewest rows); a draw with smaller n_H stops a little below n_term
    n_H = max(float(pqcd_anchor(jnp.asarray(x))[1]) for x in np.linspace(0.0, math.log(4.0), 65))
    grid = np.exp(np.linspace(math.log(n_L), math.log(n_H), cfg.n_grid_pts))
    n_sel = int(np.sum(grid <= cfg.n_term_fac * N_SAT))
    return Plan(n_keep=n_keep, n_sel=n_sel)


def bridge_table(model, theta, nep_names, plan: Plan, cfg: Config = Config()):
    """EOSTable of one draw (geometric units), differentiable in ``theta``."""
    from ..eos.base import build_h_and_cs2
    n, eps, p, _ = bridge_arrays(model, theta, nep_names, plan, cfg)
    return build_h_and_cs2(n, eps, p)


def draw_theta(rng, nep, corr_range=(0.2, 0.4), levels=9, nep_names=None):
    """A prior draw of theta for given nep (dict): corr_frac ~ U(corr_range),
    ln X ~ U(0, ln 4), z ~ N(0, 1)."""
    nep_names = list(nep) if nep_names is None else nep_names
    c = rng.uniform(*corr_range)
    lnX = rng.uniform(0.0, math.log(4.0))
    z = rng.standard_normal((2 ** levels - 1) * 3)
    return np.concatenate([[nep[k] for k in nep_names], [c, lnX], z])
