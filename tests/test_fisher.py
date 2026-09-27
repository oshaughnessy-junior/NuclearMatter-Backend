"""Validate the Fisher/sensitivity utility against a *re-rooted* fixed-mass FD.

The implicit-function-theorem sensitivities are compared to the true fixed-mass finite
difference -- for each perturbed EOS we re-solve the central enthalpy that gives the
target mass (Newton-polished) and evaluate the observable there, so the reference has
NO family-interpolation noise.  Against that truth the IFT gradients are accurate to a
few percent for all observables (the residual on Lambda is the documented single-star
tidal tangent, see test_ad_gradients).
"""
import numpy as np
import jax

from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, m_max
from nmbackend.tov.family import default_hc_grid
from nmbackend.tov.solver import solve_structure
from nmbackend.inference import (sensitivity_matrix, eos_observables,
                                 fisher_matrix, fisher_covariance)

mm = MetaModelEOS()
_P = {"L_sym": 60.0, "cs2_tail": 0.5}
_PN = ["L_sym", "cs2_tail"]
_MT = 1.4
_RT = 1e-11


def _hc_for_mass(table, m):
    fam = build_family(table, default_hc_grid(table, n_models=220), rtol=_RT,
                       atol=_RT * 1e-2)
    b = np.flatnonzero(np.asarray(fam.stable))          # neutron-star branch
    M = np.asarray(fam.M_msun)[b]; H = np.asarray(fam.h_c)[b]
    hc = float(np.interp(m, M, H))
    for _ in range(3):                                  # Newton polish onto m
        pt = solve_structure(table, hc, rtol=_RT, atol=_RT * 1e-2)
        dM = float(jax.jacfwd(lambda h: solve_structure(
            table, h, rtol=_RT, atol=_RT * 1e-2).M_msun)(hc))
        hc = hc - (float(pt.M_msun) - m) / dM
    return hc


def _rerooted_obs(p):
    """[Lambda(_MT), R(_MT), M_max] with hc re-rooted to the exact mass (no interp)."""
    t = mm.eos_table(p)
    hc = _hc_for_mass(t, _MT)
    pt = solve_structure(t, hc, rtol=_RT, atol=_RT * 1e-2)
    return np.array([float(pt.Lambda), float(pt.R_km),
                     float(m_max(build_family(t, default_hc_grid(t, 220),
                                              rtol=_RT, atol=_RT * 1e-2)))])


def _fd_col(name, dx):
    pp = dict(_P); pp[name] += dx
    pm = dict(_P); pm[name] -= dx
    return (_rerooted_obs(pp) - _rerooted_obs(pm)) / (2 * dx)


def test_sensitivities_match_rerooted_finite_difference():
    J, onames, _ = sensitivity_matrix(mm, _P, _PN, masses_lambda=(_MT,),
                                      masses_radius=(_MT,), include_mmax=True,
                                      hc_grid_n=220, rtol=_RT, atol=_RT * 1e-2)
    J = np.asarray(J)
    Jfd = np.stack([_fd_col("L_sym", 0.4), _fd_col("cs2_tail", 0.01)], axis=1)
    assert onames == ["Lambda(1.4)", "R(1.4)", "M_max"]
    # Lambda rows within 6%, R within 2%, M_max within 1.5% -- all correct sign.
    # dR/dcs2_tail: AD is 1.1% below the converged FD without a crust and 1.6%
    # below with the SLy crust (FD steps 0.0025-0.02 agree to 0.1%).
    for j in range(2):
        assert abs(J[0, j] - Jfd[0, j]) <= 0.06 * abs(Jfd[0, j])       # Lambda
        assert abs(J[1, j] - Jfd[1, j]) <= 0.02 * abs(Jfd[1, j]) + 1e-4   # R
        assert abs(J[2, j] - Jfd[2, j]) <= 0.015 * abs(Jfd[2, j]) + 1e-4  # M_max


def test_fisher_covariance_positive_definite():
    J, _, _ = sensitivity_matrix(mm, _P, _PN, masses_lambda=(_MT,),
                                 masses_radius=(_MT,), include_mmax=True,
                                 hc_grid_n=160)
    v, _ = eos_observables(mm, _P, masses_lambda=(_MT,), masses_radius=(_MT,),
                           include_mmax=True, hc_grid_n=160)
    sigma = 0.1 * np.abs(np.asarray(v))
    F = np.asarray(fisher_matrix(J, sigma))
    C = np.asarray(fisher_covariance(J, sigma))
    assert np.allclose(F, F.T)
    assert np.all(np.linalg.eigvalsh(F) > 0)
    assert np.all(np.diag(C) > 0)
