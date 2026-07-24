"""Central (core) properties along a TOV family.

Exposes the quantities the IPAD-TOV analysis needs per stellar model:
central energy density / pressure / sound speed, the central EOS parameter
X = P_c/eps_c, and the log-stability slope Psi = 2 dlnM/dln eps_c
(dM/dh_c from the family's forward-mode AD; d eps_c/dh_c by a fine finite
difference of the table's eps(h) interpolant).

Numpy outputs (post-processing layer, not differentiable end-to-end; the
underlying dM/dh_c IS the AD derivative from ``build_family``).
"""
from typing import NamedTuple

import numpy as np

__all__ = ["CentralFamily", "family_central", "central_at_mass"]


class CentralFamily(NamedTuple):
    """Central quantities along the family grid (same order as Family)."""
    h_c: np.ndarray
    eps_c: np.ndarray      # geometric km^-2
    p_c: np.ndarray        # geometric km^-2
    cs2_c: np.ndarray      # c^2
    X: np.ndarray          # P_c/eps_c
    Psi: np.ndarray        # 2 dlnM/dln eps_c
    M_msun: np.ndarray
    Lambda: np.ndarray
    stable: np.ndarray


def family_central(table, fam, dh: float = 1e-5) -> CentralFamily:
    """Central properties for every model in ``fam`` (built on ``table``)."""
    hc = np.asarray(fam.h_c, dtype=float)
    M = np.asarray(fam.M_msun, dtype=float)
    lam = np.asarray(fam.Lambda, dtype=float)
    dM_dhc = np.asarray(fam.dM_dhc, dtype=float)
    stable = np.asarray(fam.stable, dtype=bool)

    eps, p, cs2 = (np.asarray(v, dtype=float) for v in table.at_h(hc))
    eps_hi = np.asarray(table.at_h(hc + dh)[0], dtype=float)
    eps_lo = np.asarray(table.at_h(hc - dh)[0], dtype=float)
    deps_dhc = (eps_hi - eps_lo) / (2.0 * dh)
    with np.errstate(divide="ignore", invalid="ignore"):
        Psi = 2.0 * (dM_dhc / M) * (eps / deps_dhc)
    return CentralFamily(hc, eps, p, cs2, p / eps, Psi, M, lam, stable)


def central_at_mass(cf: CentralFamily, m_msun: float) -> dict:
    """Interpolate central quantities at mass ``m_msun`` on the stable branch.

    Returns a dict with h_c, eps_c, p_c, cs2_c, X, Psi, Lambda; raises
    ValueError if the stable branch does not cover the requested mass.
    """
    s = cf.stable
    if s.sum() < 4:
        raise ValueError("stable branch too short")
    M = cf.M_msun[s]
    order = np.argsort(M)
    M = M[order]
    if not (M.min() <= m_msun <= M.max()):
        raise ValueError(
            f"mass {m_msun} outside stable branch [{M.min():.3f}, {M.max():.3f}]")

    def at(v):
        return float(np.interp(m_msun, M, v[s][order]))

    return dict(
        h_c=at(cf.h_c), eps_c=at(cf.eps_c), p_c=at(cf.p_c),
        cs2_c=at(cf.cs2_c), X=at(cf.X), Psi=at(cf.Psi),
        Lambda=at(cf.Lambda),
    )
