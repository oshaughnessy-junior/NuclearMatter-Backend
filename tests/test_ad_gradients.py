"""Validate automatic differentiation against finite differences.

Forward-mode AD (jax.jacfwd) is the supported path for TOV-output sensitivities
(it reproduces the perturbation / delta-TOV equations).  Reverse-mode is checked
for structure-only (tidal=False) solves.
"""
import jax
import numpy as np
from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, m_max, lambda_of_mass
from nmbackend.tov.family import default_hc_grid
from nmbackend.tov.solver import solve_structure

mm = MetaModelEOS()
_HC = 0.20 * float(mm.eos_table().h[-1])   # absolute central enthalpy ~1.4 Msun


def _hires_family(params):
    t = mm.eos_table(params)
    return build_family(t, default_hc_grid(t, n_models=160), n_steps=800)


def test_fwd_grad_mmax_matches_fd():
    # M_max is numerically smooth -> forward-mode AD matches FD to <1%.
    def f(Ksym):
        return m_max(_hires_family({"K_sym": Ksym}))
    g = float(jax.jacfwd(f)(-100.0))
    fd = (float(f(-95.0)) - float(f(-105.0))) / 10.0
    assert abs(g - fd) <= 1e-2 * abs(fd) + 1e-4


def test_fwd_grad_radius_matches_fd():
    # Radius at fixed central enthalpy is smooth -> AD matches FD to ~1e-4.
    def f(Lsym):
        return solve_structure(mm.eos_table({"L_sym": Lsym}), _HC, n_steps=800).R_km
    g = float(jax.jacfwd(f)(60.0))
    fd = (float(f(60.5)) - float(f(59.5))) / 1.0
    assert abs(g - fd) <= 1e-3 * abs(fd) + 1e-5


def test_fwd_grad_lambda_finite_right_sign_order():
    # dLambda/dparam is finite, has the correct sign, and is order-correct vs a
    # converged FD.  Precision is currently LIMITED near the surface: the surface
    # continuation introduces a thin, stiff low-cs^2 shell that materially affects
    # Lambda (so we keep its true small cs^2 -> accurate Lambda value), but the
    # forward-mode tangent through the adaptive solver is under-resolved there
    # (~25% at rtol=1e-9).  Tightening this is roadmap action #2 (monotone,
    # thermodynamically-consistent cs^2).  See docs/roadmap.md.
    def f(Lsym):
        return solve_structure(mm.eos_table({"L_sym": Lsym}), _HC,
                               rtol=1e-9, atol=1e-11).Lambda
    g = float(jax.jacfwd(f)(60.0))
    fd = (float(f(60.5)) - float(f(59.5))) / 1.0
    assert np.isfinite(g)
    assert g * fd > 0.0                                 # correct sign
    assert abs(g - fd) <= 0.5 * abs(fd) + 1.0           # order-correct (~25% now)


def test_reverse_grad_structure_only():
    # reverse-mode is supported on the lightweight scan backend, structure-only.
    t = mm.eos_table()
    def Mof(hc):
        return solve_structure(t, hc, backend="scan", n_steps=2000,
                               tidal=False).M_msun
    g = float(jax.grad(Mof)(0.6 * float(t.h[-1])))
    assert np.isfinite(g)
