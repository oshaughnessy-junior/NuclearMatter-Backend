"""Constrained Brownian-bridge EOS prior (Gorda-Komoltsev-Kurkela-Sunde 2025).

Implements the nonparametric prior of arXiv:2512.18044: EOS draws that are
**mechanically stable, causal, and thermodynamically consistent by construction**,
bridging a low-density nuclear anchor to the perturbative-QCD limit.  No shooting,
no accept/reject against a QCD likelihood -- consistency is built into the sampler.

Method (faithful to the paper):

1. **KK 3-volume** (Komoltsev & Kurkela 2022, arXiv:2111.05350; GKKS App. A): given
   endpoint triplets beta_L=(mu_L,n_L,p_L), beta_H=(mu_H,n_H,p_H), any valid
   intermediate point lies in an analytic volume V: n_min(mu) <= n <= n_max(mu) and
   p_min(mu) <= p <= p_max(mu,n).

2. **Self-similar (fractal) refinement** (GKKS Sec. 2.2 / App. B): recursively draw a
   uniform point in V and subdivide, producing an EOS with structure on all scales
   that fills the entire allowed function space.

3. **Diffusion smoothing** (GKKS Sec. 2.3 / App. C): heat-kernel smooth mu(n) with a
   density-dependent diffusion coefficient D(n) = c^2 n^2 / 4 (flow time tau=1),
   giving a log-constant sound-speed correlation length sigma/n = c.  D ~ n^2
   saturates the causality-preservation condition D'' - D'/n <= 0, and D>0 preserves
   stability; thermodynamics are rebuilt by integrating eps = eps_L + int mu dn,
   p = mu n - eps.

Anchors: the low end comes from OUR nuclear model (metamodel band at n_L = 2 n_sat),
so the prior is anchored to this backend's physics; the high end is the partial-N3LO
pQCD point at mu_H = 2.6 GeV (Gorda et al. 2021 values as tabulated in KK 2022
Table I), with renormalization-scale parameter X in [1,4] sampled log-uniformly.

Units: mu in MeV, n in fm^-3, p and eps in MeV/fm^3.

This module is plain numpy (prior *generation* need not be differentiable); the
resulting tables feed the standard differentiable EOSTable/TOV machinery.
"""
from __future__ import annotations

import math
import numpy as np

N_SAT = 0.16                      # fm^-3
MU_H = 2600.0                     # MeV; pQCD matching point (KK 2022)

#: partial-N3LO pQCD (n_H [fm^-3], p_H [MeV/fm^3]) at mu = 2.6 GeV vs X
#: (Gorda et al. 2021 as tabulated in Komoltsev & Kurkela 2022, Table I)
PQCD_TABLE = {1.0: (6.14, 2334.0), 2.0: (6.47, 3823.0), 4.0: (6.87, 4284.0)}


def pqcd_anchor(X):
    """(mu_H, n_H, p_H) at mu=2.6 GeV, interpolated in ln X over X in [1,4]."""
    Xs = np.array(sorted(PQCD_TABLE))
    lnX = np.log(np.clip(X, Xs[0], Xs[-1]))
    ns = np.array([PQCD_TABLE[x][0] for x in Xs])
    ps = np.array([PQCD_TABLE[x][1] for x in Xs])
    # quadratic in lnX through the three tabulated values
    n_H = float(np.polyval(np.polyfit(np.log(Xs), ns, 2), lnX))
    p_H = float(np.polyval(np.polyfit(np.log(Xs), ps, 2), lnX))
    return MU_H, n_H, p_H


# ---------------------------------------------------------------- KK 3-volume ----
def kk_volume(bL, bH):
    """Closures (n_min, n_max, p_min, p_max, mu_c) for the KK volume between two
    triplets ``bL=(muL,nL,pL)``, ``bH=(muH,nH,pH)`` (GKKS App. A, eqs. A1-A7)."""
    muL, nL, pL = bL
    muH, nH, pH = bH
    dp = pH - pL
    den = muL * nH - muH * nL
    num = muL * muH * (muH * nH - muL * nL - 2.0 * dp)
    if den <= 0.0 or num <= 0.0:
        mu_c = muH if den <= 0.0 else muL
    else:
        mu_c = math.sqrt(min(max(num / den, muL ** 2), muH ** 2))

    def n_min(mu):
        if mu <= mu_c:
            return nL * mu / muL
        return (mu ** 3 * nH - mu * muH * (muH * nH - 2.0 * dp)) / \
               ((mu ** 2 - muL ** 2) * muH)

    def n_max(mu):
        if mu >= mu_c:
            return nH * mu / muH
        return (mu ** 3 * nL - muL * mu * (muL * nL + 2.0 * dp)) / \
               ((mu ** 2 - muH ** 2) * muL)

    def p_min(mu):
        return pL + (mu ** 2 - muL ** 2) / (2.0 * mu) * n_min(mu)

    n_c0 = n_max(muL) / muL          # n_c(mu) = n_c0 * mu  (eq. A7)

    def p_max(mu, n):
        if n <= n_c0 * mu:
            return pL + (mu ** 2 - muL ** 2) / (2.0 * mu) * n
        return pH - (muH ** 2 - mu ** 2) / (2.0 * mu) * n

    return n_min, n_max, p_min, p_max, mu_c


def _interval_consistent(bL, bH, rtol=1e-9):
    """KK existence condition: dp_min <= dp <= dp_max for the interval."""
    muL, nL, pL = bL
    muH, nH, pH = bH
    if muH <= muL or nH < nL:
        return False
    dp = pH - pL
    span = (muH ** 2 - muL ** 2)
    dp_lo = nL * span / (2.0 * muL)
    dp_hi = nH * span / (2.0 * muH)
    tol = rtol * max(abs(dp_hi), 1.0)
    return (dp_lo - tol) <= dp <= (dp_hi + tol)


# ------------------------------------------------ uniform sampling in the volume --
def sample_point_in_volume(rng, bL, bH, n_cdf=256):
    """Draw beta=(mu,n,p) uniformly from the KK volume (GKKS App. B).

    mu from the marginal P(mu^2) (eq. B11) via a numeric inverse-CDF (robust against
    the printed closed form); n from the triangular slice's tent-shaped marginal;
    p uniformly within [p_min, p_max(mu,n)].
    """
    muL = bL[0]; muH = bH[0]
    n_min, n_max, p_min, p_max, mu_c = kk_volume(bL, bH)

    # --- mu: P(mu^2) per eq. (B11), normalized so P(mu_c)=1
    m2 = np.linspace(muL ** 2, muH ** 2, n_cdf)
    mc2, mL2, mH2 = mu_c ** 2, muL ** 2, muH ** 2
    dens = np.where(
        m2 <= mc2,
        np.divide((m2 - mL2) * (mH2 - mc2), (mc2 - mL2) * (mH2 - m2),
                  out=np.zeros_like(m2), where=(mH2 - m2) > 0),
        np.divide((mH2 - m2) * (mc2 - mL2), (mH2 - mc2) * (m2 - mL2),
                  out=np.zeros_like(m2), where=(m2 - mL2) > 0),
    )
    dens = np.clip(dens, 0.0, None)
    dens[~np.isfinite(dens)] = 0.0         # singular slice edge (den->0): no weight
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                                           * np.diff(m2))])
    if not (np.isfinite(cdf[-1]) and cdf[-1] > 0.0):  # degenerate/singular -> midpoint
        mu = math.sqrt(0.5 * (mL2 + mH2))
    else:
        mu = math.sqrt(float(np.interp(rng.uniform(0.0, cdf[-1]), cdf, m2)))

    # --- n: tent-shaped marginal h(n) = p_max(mu,n) - p_min(mu) on [n_lo, n_hi]
    n_lo, n_hi = n_min(mu), n_max(mu)
    n_apex = min(max((n_max(muL) / muL) * mu, n_lo), n_hi)
    pmin = p_min(mu)
    h_apex = max(p_max(mu, n_apex) - pmin, 0.0)
    h_lo = max(p_max(mu, n_lo) - pmin, 0.0)
    h_hi = max(p_max(mu, n_hi) - pmin, 0.0)
    A1 = 0.5 * (h_lo + h_apex) * max(n_apex - n_lo, 0.0)
    A2 = 0.5 * (h_apex + h_hi) * max(n_hi - n_apex, 0.0)
    if A1 + A2 <= 0.0:
        n = 0.5 * (n_lo + n_hi)
    elif rng.uniform(0.0, A1 + A2) < A1:
        # rising trapezoid on [n_lo, n_apex]
        u = rng.uniform()
        if h_apex > h_lo:
            n = n_lo + (n_apex - n_lo) * (
                (math.sqrt(h_lo ** 2 + u * (h_apex ** 2 - h_lo ** 2)) - h_lo)
                / (h_apex - h_lo))
        else:
            n = n_lo + u * (n_apex - n_lo)
    else:
        # falling trapezoid on [n_apex, n_hi]
        u = rng.uniform()
        if h_apex > h_hi:
            n = n_hi - (n_hi - n_apex) * (
                (math.sqrt(h_hi ** 2 + u * (h_apex ** 2 - h_hi ** 2)) - h_hi)
                / (h_apex - h_hi))
        else:
            n = n_apex + u * (n_hi - n_apex)

    # --- p: uniform within the slice
    p = rng.uniform(pmin, max(p_max(mu, n), pmin))
    return (mu, float(n), float(p))


# ------------------------------------------------------- fractal refinement ------
def fractal_refine(rng, bL, bH, levels=9, min_dlnmu=1e-4):
    """Self-similar refinement (GKKS Sec. 2.2): recursively bisect [bL, bH] with
    uniform draws from each sub-volume.  Returns the sorted list of triplets."""
    pts = []

    def recurse(a, b, depth):
        if depth <= 0 or math.log(b[0] / a[0]) < min_dlnmu:
            return
        if not _interval_consistent(a, b):
            return
        try:
            m = sample_point_in_volume(rng, a, b)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return
        # enforce strict ordering (numerical safety)
        if not (a[0] < m[0] < b[0] and a[1] <= m[1] <= b[1]
                and a[2] <= m[2] <= b[2]):
            return
        pts.append(m)
        recurse(a, m, depth - 1)
        recurse(m, b, depth - 1)

    recurse(tuple(bL), tuple(bH), levels)
    return [tuple(bL)] + sorted(pts) + [tuple(bH)]


# ------------------------------------------------------------ diffusion ----------
def _thomas(a, b, c, d):
    """Tridiagonal solve (a: sub, b: diag, c: super, d: rhs)."""
    n = len(d)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def diffuse_mu_of_n(n_grid, mu, corr_frac, n_steps=40):
    """Heat-kernel smooth the EOS: constant-D diffusion of ln(mu) vs x = ln(n).

    Equivalent to the paper's D(n) = corr_frac^2 n^2 / 4 choice (GKKS Sec. 2.3.1:
    sigma/n = corr_frac, log-constant correlation length), but formulated in
    (ln mu, ln n) where it is *provably safe*: with constant D the sound speed
    cs^2 = d(ln mu)/d(ln n) satisfies the SAME heat equation, so it obeys a maximum
    principle -- it can never leave the initial range [0, 1].  Causality and
    stability are preserved exactly (continuum), and to discretization accuracy on
    the uniform x grid (backward Euler is an M-matrix scheme; the discrete derivative
    satisfies the same update away from the two boundary cells).

    Dirichlet ends pin ln(mu) at both anchors, so mu_L, mu_H are exact and
    int cs^2 dx = ln(mu_H/mu_L) is conserved identically.
    """
    n = np.asarray(n_grid, float)
    x = np.log(n)
    if not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-6):
        raise ValueError("diffuse_mu_of_n requires a log-uniform density grid")
    dx = x[1] - x[0]
    u = np.log(np.array(mu, float))
    N = x.size
    D = 0.25 * corr_frac ** 2                          # sigma_x = corr_frac
    dt = 1.0 / n_steps
    r = dt * D / dx ** 2
    a = np.full(N, -r); b = np.full(N, 1.0 + 2.0 * r); c = np.full(N, -r)
    a[0] = c[0] = a[-1] = c[-1] = 0.0
    b[0] = b[-1] = 1.0                                 # Dirichlet ends
    for _ in range(n_steps):
        u = _thomas(a.copy(), b.copy(), c.copy(), u.copy())
    return np.exp(u)


# ------------------------------------------------------------- full draw ---------
def draw_bridge_eos(rng, low_table=None, model=None, nep_params=None,
                    corr_frac=0.2, levels=9, X=None, n_L_fac=2.0,
                    n_term_fac=12.0, n_grid_pts=600, n_steps=40,
                    n_table_pts=800):
    """One stable-causal-consistent EOS draw bridging nuclear matter to pQCD.

    Parameters
    ----------
    rng        : numpy Generator.
    low_table / model / nep_params : low-density anchor.  Either give ``low_table``
        (arrays dict with n, eps, p in fm^-3 / MeV/fm^3) or a ``model`` (EOSModel,
        default MetaModelEOS) with parameter dict ``nep_params`` drawn outside.
    corr_frac  : sound-speed correlation length as a fraction of density
        (sigma/n; GKKS use a hierarchical U[0.2, 0.4]).
    levels     : fractal refinement depth (2^levels - 1 interior points max).
    X          : pQCD renormalization-scale parameter (log-uniform [1,4] if None).
    n_L_fac    : bridge start in units of n_sat (paper: 2).
    n_term_fac : truncation density for the returned stellar-structure table.
    n_table_pts : node count of the returned table, which adds nodes to the
        raw (n, eps, p) (their count varies by a few per draw) with
        :func:`nmbackend.eos.base.refine_table`; None keeps the raw nodes.

    Returns dict with the combined (n, eps, p) arrays, the EOSTable, the anchor
    metadata, and the endpoint pressure mismatch diagnostic.
    """
    n_L = n_L_fac * N_SAT
    # ---- low-density anchor from our nuclear model
    if low_table is None:
        if model is None:
            from ..eos.metamodel import MetaModelEOS
            model = MetaModelEOS()
        t = model.eos_table(nep_params or {})
        n_lo = np.asarray(t.n); eps_lo = np.asarray(t.eps_mevfm3)
        p_lo = np.asarray(t.p_mevfm3)
    else:
        n_lo = np.asarray(low_table["n"]); eps_lo = np.asarray(low_table["eps"])
        p_lo = np.asarray(low_table["p"])
    keep = n_lo < n_L
    n_lo, eps_lo, p_lo = n_lo[keep], eps_lo[keep], p_lo[keep]
    # anchor point at exactly n_L (interpolate the model)
    eps_L = float(np.interp(n_L, np.concatenate([n_lo, [n_L * 1.0001]]),
                            np.concatenate([eps_lo, [eps_lo[-1]
                                            * (n_L / n_lo[-1])]])))
    p_L = float(np.interp(n_L, n_lo, p_lo)) if n_lo[-1] >= n_L else float(p_lo[-1])
    eps_L = float(np.interp(n_L, n_lo, eps_lo)) if n_lo[-1] >= n_L else eps_L
    mu_L = (eps_L + p_L) / n_L
    bL = (mu_L, n_L, p_L)

    # ---- pQCD anchor
    if X is None:
        X = float(np.exp(rng.uniform(np.log(1.0), np.log(4.0))))
    bH = pqcd_anchor(X)

    if not _interval_consistent(bL, bH):
        raise ValueError(f"anchors inconsistent (mu_L={mu_L:.1f} MeV); "
                         "check the low-density model")

    # ---- fractal bridge + log-linear completion onto a regular ln-n grid
    pts = fractal_refine(rng, bL, bH, levels=levels)
    mu_pts = np.array([q[0] for q in pts])
    n_pts = np.array([q[1] for q in pts])
    grid = np.exp(np.linspace(math.log(n_L), math.log(bH[1]), n_grid_pts))
    # piecewise log-linear mu(n): chords have cs^2 in [0,1] automatically
    mu_grid = np.exp(np.interp(np.log(grid), np.log(n_pts), np.log(mu_pts)))

    # ---- diffusion smoothing
    mu_s = diffuse_mu_of_n(grid, mu_grid, corr_frac, n_steps=n_steps)
    mu_s = np.maximum.accumulate(mu_s)                # numerical monotone guard

    # ---- rebuild thermodynamics: eps = eps_L + int mu dn ; p = mu n - eps
    eps_b = eps_L + np.concatenate(
        [[0.0], np.cumsum(0.5 * (mu_s[1:] + mu_s[:-1]) * np.diff(grid))])
    p_b = mu_s * grid - eps_b
    p_end_mismatch = (p_b[-1] - bH[2]) / bH[2]

    # ---- combine with the low-density table and truncate for stellar structure
    n_term = n_term_fac * N_SAT
    sel = grid <= n_term
    n_all = np.concatenate([n_lo, grid[sel]])
    eps_all = np.concatenate([eps_lo, eps_b[sel]])
    p_all = np.concatenate([p_lo, p_b[sel]])
    # lazy import: keeps this module importable without the (JAX) table machinery
    from ..eos.base import refine_table
    from ..eos.compose import eos_table_from_arrays
    if n_table_pts is None:
        table = eos_table_from_arrays(n_all, eps_all, p_all, n_surface=None)
    else:       # NumPy until the length is fixed: no compile per raw length
        table = refine_table(eos_table_from_arrays(
            n_all, eps_all, p_all, n_surface=None, xp=np), n_table_pts)

    cs2 = np.diff(np.log(mu_s)) / np.diff(np.log(grid))
    return dict(table=table, n=n_all, eps=eps_all, p=p_all,
                bridge=dict(n=grid, mu=mu_s, mu0=mu_grid, p=p_b, eps=eps_b,
                            cs2=cs2),
                meta=dict(X=X, corr_frac=corr_frac, mu_L=mu_L, p_L=p_L,
                          n_L=n_L, p_end_mismatch=float(p_end_mismatch)))


def _pbar(total, desc):
    """tqdm if available, else a quiet stderr ticker; .update(k)/.close()."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="eos")
    except Exception:
        import sys

        class _T:
            def __init__(self):
                self.k = 0

            def update(self, j=1):
                self.k += j
                if self.k % max(1, total // 20) == 0 or self.k == total:
                    print(f"  {desc}: {self.k}/{total}", file=sys.stderr,
                          flush=True)

            def close(self):
                pass
        return _T()


def draw_prior_ensemble(n_draws, seed=0, corr_range=(0.2, 0.4),
                        nep_sampler=None, progress=False, **kw):
    """Draw an ensemble; returns (tables, records).  Hierarchical corr_frac per
    GKKS Sec. 4 (uniform over ``corr_range``); NEPs via ``nep_sampler(rng)``.
    ``progress=True`` shows a bar (tqdm if installed)."""
    rng = np.random.default_rng(seed)
    bar = _pbar(n_draws, "bridge draws") if progress else None
    tables, records = [], []
    attempts = 0
    while len(tables) < n_draws and attempts < 20 * n_draws:
        attempts += 1
        c = float(rng.uniform(*corr_range))
        nep = nep_sampler(rng) if nep_sampler else {}
        try:
            d = draw_bridge_eos(rng, nep_params=nep, corr_frac=c, **kw)
        except (ValueError, FloatingPointError, OverflowError):
            continue
        tables.append(d["table"])
        rec = dict(corr_frac=c, **{f"nep_{k}": v for k, v in nep.items()})
        rec.update({k: d["meta"][k] for k in ("X", "p_end_mismatch")})
        records.append(rec)
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return tables, records
