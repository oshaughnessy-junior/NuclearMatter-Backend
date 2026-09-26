"""Nucleonic direct-Urca (dUrca) onset from a beta-equilibrium composition.

The process n -> p + l + nubar (l = e, mu) conserves momentum at the Fermi
surfaces only if the Fermi momenta close a triangle, k_Fn <= k_Fp + k_Fl
(Lattimer, Pethick, Prakash & Haensel, PRL 66, 2701 (1991)).  With
k_Fi = (3 pi^2 x_i n)^(1/3), charge neutrality and npe matter this is
x_p >= 1/9; muons lower the threshold toward ~0.148 at high density
(the upper end of the npe mu band quoted in LPPH).

Conventions for "no onset":
  * ``n_DU = inf`` if the triangle never closes on the composition grid.
  * ``M_DU = inf`` if n_DU is above the central density of the maximum-mass
    star (no stable star runs dUrca), including n_DU = inf.
  NaN is not used as a sentinel; it means a failed solve.

Numpy post-processing layer (not differentiable end-to-end).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .tov.solver import solve_structure

__all__ = ["DUrcaOnset", "fermi_momenta", "durca_onset", "durca_mass",
           "NPE_THRESHOLD"]

#: npe-matter threshold proton fraction: k_Fn = k_Fp + k_Fe with x_e = x_p
NPE_THRESHOLD = 1.0 / 9.0

_3PI2 = 3.0 * math.pi ** 2


class DUrcaOnset(NamedTuple):
    """dUrca onset for one composition (densities in fm^-3)."""
    n_DU: float        # electron channel onset density (inf if none)
    x_p_DU: float      # proton fraction at n_DU (nan if none)
    n_DU_mu: float     # muon channel onset density (inf if none)
    x_p_DU_mu: float   # proton fraction at n_DU_mu (nan if none)


def fermi_momenta(comp):
    """(k_Fn, k_Fp, k_Fe, k_Fmu) in fm^-1 from a Composition."""
    n = np.asarray(comp.n, float)
    xp = np.asarray(comp.x_p, float)
    kf = lambda x: np.cbrt(_3PI2 * np.clip(x, 0.0, None) * n)
    return (kf(1.0 - xp), kf(xp), kf(np.asarray(comp.x_e, float)),
            kf(np.asarray(comp.x_mu, float)))


def _first_crossing(n, f, xp, active=None):
    """Lowest n where f = k_Fp + k_Fl - k_Fn goes from < 0 to >= 0.

    ``active`` masks points where the lepton is present (muons); the onset is
    only accepted at such points.  Linear interpolation between grid nodes.
    """
    ok = np.ones_like(f, bool) if active is None else active
    if ok[0] and f[0] >= 0.0:
        return float(n[0]), float(xp[0])       # open from the grid start
    for i in range(1, len(n)):
        if ok[i] and f[i] >= 0.0:
            if not ok[i - 1] or f[i - 1] >= 0.0:   # opens as the lepton appears
                return float(n[i]), float(xp[i])
            t = -f[i - 1] / (f[i] - f[i - 1])
            return (float(n[i - 1] + t * (n[i] - n[i - 1])),
                    float(xp[i - 1] + t * (xp[i] - xp[i - 1])))
    return math.inf, math.nan


def durca_onset(comp, n_min=0.08) -> DUrcaOnset:
    """Onset densities of the electron and muon dUrca channels.

    Only densities n >= ``n_min`` (fm^-3) are searched.  The default sits near
    the crust-core transition: below it uniform npe matter is not the ground
    state, and a Taylor-expanded S(n) can give spurious large x_p there.
    Only the first opening is reported; a re-closing at higher density is
    ignored.  Resolution is that of the composition grid (linear in n).
    """
    kn, kp, ke, kmu = fermi_momenta(comp)
    n = np.asarray(comp.n, float)
    xp = np.asarray(comp.x_p, float)
    s = n >= n_min
    n, xp, kn, kp, ke, kmu = (a[s] for a in (n, xp, kn, kp, ke, kmu))
    n_e, x_e = _first_crossing(n, kp + ke - kn, xp)
    n_m, x_m = _first_crossing(n, kp + kmu - kn, xp, active=kmu > 0.0)
    return DUrcaOnset(n_e, x_e, n_m, x_m)


def durca_mass(table, n_DU, h_max_stable=None, **tov_kw):
    """Gravitational mass (Msun) of the star whose central density is n_DU.

    ``table`` : EOSTable of the same model (its ``n`` grid maps n -> h).
    ``h_max_stable`` : central pseudo-enthalpy of the maximum-mass star, e.g.
        ``fam.h_c[argmax(where(fam.stable, fam.M_msun, -inf))]``.  If omitted,
        n_DU is only checked against the table range and the caller owns the
        stability check.
    Returns inf when n_DU is inf or above the maximum-mass central density.
    Extra keywords go to :func:`nmbackend.tov.solver.solve_structure`.
    """
    if not np.isfinite(n_DU):
        return math.inf
    n = np.asarray(table.n, float)
    h = np.asarray(table.h, float)
    if n_DU > n[-1]:
        return math.inf
    h_DU = float(np.interp(math.log(n_DU), np.log(n), h))
    if h_max_stable is not None and h_DU > float(h_max_stable):
        return math.inf
    return float(solve_structure(table, h_DU, **tov_kw).M_msun)
