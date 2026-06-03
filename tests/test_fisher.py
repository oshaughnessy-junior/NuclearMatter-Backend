"""Validate the Fisher/sensitivity utility against finite differences.

The implicit-function-theorem sensitivities must match FD: tightly for the smooth
observables (R(m), M_max) and at least in sign + few-x-10% for Lambda(m) (whose
single-star tidal tangent carries a known small residual, see test_ad_gradients).
"""
import numpy as np
import pytest

from nmbackend.eos import MetaModelEOS
from nmbackend.inference import (sensitivity_matrix, eos_observables,
                                 fisher_matrix, fisher_covariance)

mm = MetaModelEOS()
_P = {"L_sym": 60.0, "cs2_tail": 0.5}
_PN = ["L_sym", "cs2_tail"]
_KW = dict(masses_lambda=(1.4,), masses_radius=(1.4,), include_mmax=True,
           hc_grid_n=140)


def _obs(p):
    v, _ = eos_observables(mm, p, **_KW)
    return np.asarray(v)


def _fd_col(name, dx):
    pp = dict(_P); pp[name] += dx
    pm = dict(_P); pm[name] -= dx
    return (_obs(pp) - _obs(pm)) / (2 * dx)


def test_sensitivities_match_finite_difference():
    J, onames, pnames = sensitivity_matrix(mm, _P, _PN, **_KW)
    J = np.asarray(J)
    Jfd = np.stack([_fd_col("L_sym", 0.5), _fd_col("cs2_tail", 0.01)], axis=1)
    assert onames == ["Lambda(1.4)", "R(1.4)", "M_max"]
    iL, iR, iM = 0, 1, 2
    # R and M_max: tight
    for i in (iR, iM):
        for j in range(2):
            assert abs(J[i, j] - Jfd[i, j]) <= 0.03 * abs(Jfd[i, j]) + 1e-4
    # Lambda: correct sign and within 15% (incl. FD-reference noise) -- crucially the
    # IFT fixes the family-interpolation failure that flipped dLambda/dcs2_tail's sign
    for j in range(2):
        assert J[iL, j] * Jfd[iL, j] > 0
        assert abs(J[iL, j] - Jfd[iL, j]) <= 0.15 * abs(Jfd[iL, j])


def test_fisher_covariance_positive_definite():
    J, _, _ = sensitivity_matrix(mm, _P, _PN, **_KW)
    v, _ = eos_observables(mm, _P, **_KW)
    sigma = 0.1 * np.abs(np.asarray(v))
    F = np.asarray(fisher_matrix(J, sigma))
    C = np.asarray(fisher_covariance(J, sigma))
    assert np.allclose(F, F.T)
    assert np.all(np.linalg.eigvalsh(F) > 0)            # positive definite
    assert np.all(np.diag(C) > 0)                       # finite forecast variances
