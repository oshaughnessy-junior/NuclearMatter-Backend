"""IPAD-TOV closed-form scalings (pure numpy; no JAX dependency).

Implements the "intrinsic tidal response" chain of Shi, Cai, Li & Ma,
arXiv:2606.21402 (built on the IPAD-TOV framework of Cai & Li, EPJA 61, 55
(2025), arXiv:2501.18676):

    X   = P_c / eps_c                        central EOS parameter
        = (1/eps_c) int_0^eps_c cs^2 d eps   (energy-density-averaged cs^2)
    Psi = 2 dlnM / dln eps_c                 log-stability slope
    s_c^2(X, Psi)                            Eq. (5)
    xi(X, Psi) = 2 Pi_c                      Eq. (6)  (xi = M/R compactness)
    k2(X, Psi)                               Eq. (7) with (C1)-(C4)
    D(X, Psi) = 2 k2 / (3 xi^5)              intrinsic tidal response

CONVENTION TRAP: Eq. (7) and the appendix-C polynomials are written in
Pi_c = xi/2, while D uses the full compactness xi.  Getting this wrong
reproduces nothing; the anchor tests in tests/test_ipad.py pin the
published GW170817 mapping (X = 0.185/0.16/0.133 -> Lambda_1.4 ~
190/302/580 with Psi = 2.66) and guard the convention.

The linear calibrations quoted by the source papers (their ensembles):
    Lambda_1.4 ~ 6.41 D + 23.50      (R^2 = 0.961)          Eq. (8)
    Lambda_TOV ~ 2.36 D(X,0) + 3.26  (R^2 = 0.958)          Eq. (9)
These are ensemble-dependent — on broad nonparametric priors the linear
fit degrades substantially (see matter-backend-papers p4) — but D itself
remains a useful derived coordinate; that assessment is the point of the
paper-4 analysis, not an endorsement of the fits.

All functions broadcast over numpy arrays.
"""
import numpy as np

__all__ = [
    "sc2_ipad", "xi_ipad", "k2_ipad", "D_ipad",
    "lam14_linear", "lamtov_linear", "X_CAUSAL_CLAIMED",
]

#: Claimed causal bound on X from s_c^2(X, Psi~2.66) <= 1 [2606.21402].
#: NOT ensemble-agnostic: constant-cs2-tail and PT EOS exceed it while
#: remaining causal (p4 L0/L1 probes).
X_CAUSAL_CLAIMED = 0.381


def sc2_ipad(X, Psi):
    """Central sound speed squared s_c^2(X, Psi), 2606.21402 Eq. (5)."""
    X = np.asarray(X, dtype=float)
    Psi = np.asarray(Psi, dtype=float)
    a = 1.0 + (1.0 + Psi) / 3.0 * (1.0 + 3.0 * X**2 + 4.0 * X) / (1.0 - 3.0 * X**2)
    b = 1.0 - 24.0 * (1.0 + Psi) ** 2 / (4.0 + Psi) * X / (
        50.0 + 37.0 * Psi + 5.0 * Psi**2)
    return X * a * b


def xi_ipad(X, Psi):
    """Compactness xi = M/R = 2 Pi_c, 2606.21402 Eq. (6)."""
    X = np.asarray(X, dtype=float)
    Psi = np.asarray(Psi, dtype=float)
    lead = 2.0 * X / (1.0 + 3.0 * X**2 + 4.0 * X)
    corr = 1.0 + 36.0 * (1.0 + Psi) * X / (50.0 + 37.0 * Psi + 5.0 * Psi**2)
    return lead * corr


def k2_ipad(X, Psi):
    """Quadrupolar tidal Love number k2(X, Psi), Eq. (7) + (C1)-(C4).

    Returns nan where the expansion leaves its domain (1 - 4 Pi_c <= 0,
    i.e. xi >= 1/2).
    """
    sc2 = sc2_ipad(X, Psi)
    Pi = 0.5 * xi_ipad(X, Psi)
    X = np.asarray(X, dtype=float)
    Y = 2.0 - (2.0 * X / 7.0) / (1.0 + 3.0 * X) * (
        (1.0 + 36.0 * X + 99.0 * X**2) / (1.0 + 4.0 * X + 3.0 * X**2)
        + 3.0 / sc2)
    H1 = 2.0 - Y + 4.0 * Pi * (Y - 1.0)
    H = 13.0 - 11.0 * Y + 2.0 * Pi * (3.0 * Y - 2.0) + 8.0 * Pi**2 * (1.0 + Y)
    H2 = 12.0 * Pi * (2.0 - Y + 2.0 * Pi * (5.0 * Y - 8.0)) + 32.0 * Pi**3 * H
    one4 = 1.0 - 4.0 * Pi
    with np.errstate(invalid="ignore", divide="ignore"):
        k2 = np.where(
            one4 > 0.0,
            (256.0 / 5.0) * Pi**5 * one4**2
            / (H2 / H1 + 3.0 * one4**2 * np.log(np.where(one4 > 0, one4, 1.0))),
            np.nan,
        )
    return k2


def D_ipad(X, Psi):
    """Intrinsic tidal response D = 2 k2 / (3 xi^5)."""
    xi = xi_ipad(X, Psi)
    return (2.0 / 3.0) * k2_ipad(X, Psi) / xi**5


def lam14_linear(D):
    """Their canonical-NS calibration Lambda_1.4(D), Eq. (8). Ensemble-bound."""
    return 6.41 * np.asarray(D, dtype=float) + 23.50


def lamtov_linear(D_tov):
    """Their TOV-NS calibration Lambda_TOV(D), Eq. (9). Ensemble-bound."""
    return 2.36 * np.asarray(D_tov, dtype=float) + 3.26
