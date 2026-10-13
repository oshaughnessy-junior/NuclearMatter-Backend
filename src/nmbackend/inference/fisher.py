"""Forward-mode sensitivities and Fisher-matrix forecasting of NS observables.

The whole point of the differentiable backend is that gradients of stellar
observables with respect to EOS parameters -- dLambda/dtheta, dR/dtheta,
dM_max/dtheta -- come out by automatic differentiation, reproducing the perturbation
("delta-TOV") equations.  This module packages those into:

* :func:`eos_observables`   -- (Lambda(m_i), R(m_j), M_max) for a parameter dict;
* :func:`sensitivity_matrix`-- the Jacobian J[obs, param] by forward-mode AD;
* :func:`fisher_matrix`     -- F = J^T C^-1 J for a (diagonal) data covariance;
* :func:`fisher_covariance` -- F^-1, the forecast parameter covariance.

Fixed-mass sensitivities via the implicit-function theorem
----------------------------------------------------------
A naive ``d/dtheta`` of an interpolated ``Lambda(m_target)`` over a TOV family is
inaccurate -- the dominant term is the reparameterization of the central enthalpy as
the fixed-mass target shifts, and differentiating through the coarse family
interpolation gets it badly wrong (it can even flip the sign).  Instead we use the
implicit-function theorem on the *single-star* tangents, which are accurate at
moderate tolerance (see tests/test_ad_gradients.py):

    h_c*(theta) solves   M(h_c*, theta) = m_target
    dh_c*/dtheta       = -(dM/dtheta|_hc) / (dM/dh_c)
    dLambda/dtheta|_m  =  dLambda/dtheta|_hc + (dLambda/dh_c) * dh_c*/dtheta
    dR/dtheta|_m       =  dR/dtheta|_hc      + (dR/dh_c)      * dh_c*/dtheta

and the *envelope theorem* for the maximum mass (dM/dh_c = 0 at the peak):

    dM_max/dtheta      =  dM/dtheta|_{h_c,max}

All partials are clean forward-mode single-star derivatives.  This resolves the
"gradient tightening" backlog item for Fisher/forecasting use without needing ever
tighter ODE tolerances.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..tov import build_family, m_max, lambda_of_mass, radius_of_mass
from ..tov.family import default_hc_grid
from ..tov.solver import solve_structure

#: default solver tolerances for *gradient* work (single-star tangents).
#: Deliberately tighter than solve_structure's 1e-7: parameter tangents are
#: accurate there (~3e-5) but the h_c tangent sits at ~1e-4 (frozen-start
#: near-centre resolution), ~5e-8 at 1e-10.  Do not loosen to 1e-7; see
#: docs/devlog/2026-10-12-tov-jump-ts.md.
GRADIENT_RTOL = 1e-10
GRADIENT_ATOL = 1e-12


def eos_observables(model, params, *, masses_lambda=(1.4,), masses_radius=(1.4,),
                    include_mmax=True, hc_grid_n=160, rtol=GRADIENT_RTOL,
                    atol=GRADIENT_ATOL):
    """Stack of NS observables for one parameter dict, plus their names.

    Observables: ``Lambda(m)`` for each ``masses_lambda``, ``R(m)`` for each
    ``masses_radius`` (km), then ``M_max`` (Msun) if ``include_mmax``.  Built from a
    single TOV family (so all share one solve), branch-aware on the stable rising
    branch.  Fully forward-mode differentiable in ``params``.
    """
    table = model.eos_table(params)
    fam = build_family(table, default_hc_grid(table, n_models=hc_grid_n),
                       rtol=rtol, atol=atol)
    vals, names = [], []
    for m in masses_lambda:
        vals.append(lambda_of_mass(fam, m)); names.append(f"Lambda({m})")
    for m in masses_radius:
        vals.append(radius_of_mass(fam, m)); names.append(f"R({m})")
    if include_mmax:
        vals.append(m_max(fam)); names.append("M_max")
    return jnp.stack(vals), names


def _star_MRL(model, params, hc, rtol, atol):
    pt = solve_structure(model.eos_table(params), hc, rtol=rtol, atol=atol)
    return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])


def _partials_at(model, params, param_names, hc, rtol, atol):
    """At fixed central enthalpy ``hc``: values [M,R,Lam], d[M,R,Lam]/dtheta (3,np),
    and d[M,R,Lam]/dhc (3,) -- all clean single-star forward-mode tangents."""
    theta0 = jnp.array([float(params[k]) for k in param_names])

    def g_theta(theta):
        p = dict(params)
        for k, v in zip(param_names, theta):
            p[k] = v
        return _star_MRL(model, p, hc, rtol, atol)

    val = g_theta(theta0)
    d_theta = jax.jacfwd(g_theta)(theta0)                 # (3, n_param)
    d_hc = jax.jacfwd(lambda h: _star_MRL(model, params, h, rtol, atol))(hc)  # (3,)
    return np.asarray(val), np.asarray(d_theta), np.asarray(d_hc)


def _locate(model, params, masses, hc_grid_n, rtol_locate=1e-8):
    """Central enthalpies giving each target mass on the stable rising branch, and the
    enthalpy at maximum mass.  Values only (used as expansion points)."""
    t = model.eos_table(params)
    fam = build_family(t, default_hc_grid(t, n_models=hc_grid_n),
                       rtol=rtol_locate, atol=rtol_locate * 1e-2)
    b = np.flatnonzero(np.asarray(fam.stable))    # minimum-mass star to M_max
    M = np.asarray(fam.M_msun)[b]
    H = np.asarray(fam.h_c)[b]
    imax = int(np.argmax(M))
    hc_for = {m: float(np.interp(m, M[:imax + 1], H[:imax + 1])) for m in masses}
    return hc_for, float(H[imax])


def sensitivity_matrix(model, params, param_names, *, masses_lambda=(1.4,),
                       masses_radius=(1.4,), include_mmax=True, hc_grid_n=160,
                       rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL):
    """Jacobian J[i, j] = d(observable_i)/d(param_j) at fixed mass, via the IFT.

    Observables (rows): ``Lambda(m)`` for each ``masses_lambda``, ``R(m)`` for each
    ``masses_radius``, then ``M_max``.  Fixed-mass rows use the implicit-function
    theorem; ``M_max`` uses the envelope theorem -- both built from accurate
    single-star tangents (NOT differentiation through the family interpolation).

    ``params`` must give a numeric value for every name in ``param_names``.  Returns
    ``(J, obs_names, param_names)``, ``J`` shape ``(n_obs, n_param)``.
    """
    param_names = list(param_names)
    masses = sorted(set(masses_lambda) | set(masses_radius))
    hc_for, hc_max = _locate(model, params, masses, hc_grid_n)

    # cache single-star partials at each needed expansion point
    cache = {}
    for m in masses:
        cache[m] = _partials_at(model, params, param_names, hc_for[m], rtol, atol)

    def fixed_mass_row(m, channel):
        _, d_theta, d_hc = cache[m]
        dM_dhc = d_hc[0]
        dhc_dtheta = -d_theta[0] / dM_dhc                 # IFT
        return d_theta[channel] + d_hc[channel] * dhc_dtheta

    rows, names = [], []
    for m in masses_lambda:
        rows.append(fixed_mass_row(m, 2)); names.append(f"Lambda({m})")
    for m in masses_radius:
        rows.append(fixed_mass_row(m, 1)); names.append(f"R({m})")
    if include_mmax:
        _, d_theta_max, _ = _partials_at(model, params, param_names, hc_max, rtol, atol)
        rows.append(d_theta_max[0]); names.append("M_max")    # envelope theorem
    return np.stack(rows, axis=0), names, param_names


def fisher_matrix(J, sigma):
    """Fisher information ``F = J^T C^-1 J`` for a diagonal data covariance.

    ``sigma`` is the per-observable 1-sigma measurement uncertainty (same order as the
    rows of ``J``).  ``F`` has shape ``(n_param, n_param)``.
    """
    J = jnp.asarray(J)
    sigma = jnp.asarray(sigma)
    Cinv = jnp.diag(1.0 / sigma ** 2)
    return J.T @ Cinv @ J


def fisher_covariance(J, sigma, prior_precision=None):
    """Forecast parameter covariance ``(F + Pi)^-1`` (Pi = optional prior precision).

    The Cramer-Rao parameter covariance for a Gaussian likelihood; add a prior
    precision matrix ``prior_precision`` (n_param x n_param) to regularize / include
    priors.  Returns the covariance matrix.
    """
    F = fisher_matrix(J, sigma)
    if prior_precision is not None:
        F = F + jnp.asarray(prior_precision)
    return jnp.linalg.inv(F)
