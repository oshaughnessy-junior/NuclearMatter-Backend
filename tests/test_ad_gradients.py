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
    # Radius gradient at fixed central enthalpy.  0.1-0.2% achieved on jax 0.7.1
    # and 0.11.2 (was -2.4% on jax 0.7.1 before the where-floor fix in
    # hermite_eos_at_h; see test_jacfwd_matches_jvp_with_tidal).
    def f(Lsym):
        return solve_structure(mm.eos_table({"L_sym": Lsym}), _HC,
                               rtol=1e-9, atol=1e-11).R_km
    g = float(jax.jacfwd(f)(60.0))
    fd = (float(f(60.5)) - float(f(59.5))) / 1.0
    assert abs(g - fd) <= 5e-3 * abs(fd) + 1e-5


def test_fwd_grad_lambda_converges_with_tolerance():
    # dLambda/dL_sym converges to FD as rtol tightens (jax 0.7.1: 1e-8 +5.7%,
    # 1e-10 +0.01%, 1e-12 +0.02%), so gradient work wants rtol <~ 1e-10.
    def f(Lsym, rt):
        return solve_structure(mm.eos_table({"L_sym": Lsym}), _HC,
                               rtol=rt, atol=rt * 1e-2).Lambda
    fd = (float(f(60.2, 1e-12)) - float(f(59.8, 1e-12))) / 0.4
    g_loose = float(jax.jacfwd(lambda L: f(L, 1e-8))(60.0))
    g_tight = float(jax.jacfwd(lambda L: f(L, 1e-12))(60.0))
    assert np.isfinite(g_tight) and g_tight * fd > 0.0
    assert abs(g_tight - fd) < abs(g_loose - fd)
    assert abs(g_tight - fd) <= 0.01 * abs(fd)


def test_reverse_grad_structure_only():
    # reverse-mode is supported on the lightweight scan backend, structure-only.
    t = mm.eos_table()
    def Mof(hc):
        return solve_structure(t, hc, backend="scan", n_steps=2000,
                               tidal=False).M_msun
    g = float(jax.grad(Mof)(0.6 * float(t.h[-1])))
    assert np.isfinite(g)


def test_jacfwd_matches_jvp_with_tidal():
    # jax.jacfwd is vmap(jvp), so both must give the same tangent.  Under
    # jaxlib <= 0.9 a maximum() floor in hermite_eos_at_h made XLA miscompile
    # the batched tangent once the tidal sector was on: jacfwd(M) disagreed
    # with jvp(M), and jvp matched finite differences.
    t = mm.eos_table()
    def f(hc):
        return solve_structure(t, hc, backend="scan", n_steps=4000).M_msun
    g_fwd = float(jax.jacfwd(f)(_HC))
    g_jvp = float(jax.jvp(f, (_HC,), (1.0,))[1])
    assert abs(g_fwd - g_jvp) <= 1e-10 * abs(g_jvp)
