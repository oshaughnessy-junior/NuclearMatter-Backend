"""The fixed-step scan backend agrees with the adaptive diffrax backend.

Before 2026-10-14 the scan stepped uniformly in h from h_c - 1e-8 (h_c - h_min)
and its first RK4 step overshot r ~ sqrt(h_c - h) by orders of magnitude: at
n_steps=2000, M was +60..+210% off and Lambda NaN at almost every h_c on all
four tables here, and its reverse-mode gradients were NaN or off by factors up
to 37 (docs/devlog/2026-10-14-tov-scan-accuracy.md).  Every check below fails
on that scan.
"""
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.eos import MetaModelEOS, RMFTEOS, SkyrmeEOS
from nmbackend.tov.solver import solve_structure

_HCS = jnp.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


def _sibling(name):
    """Import tests/<name>.py by path, independent of pytest's import mode."""
    path = pathlib.Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_nmb_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bridge():
    from nmbackend.priors.bridge import draw_bridge_eos
    return draw_bridge_eos(np.random.default_rng(1), corr_frac=0.3)["table"]


# bridge1 has a density jump at fixed p (a junction-bubble cell)
_EOS = {"MetaModel": lambda: MetaModelEOS().eos_table(),
        "GM1": lambda: RMFTEOS().eos_table(),
        "SLy4": lambda: SkyrmeEOS().eos_table(), "bridge1": _bridge}


def _mrl(pt):
    return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])


_ref = jax.jit(jax.vmap(
    lambda t, hc: _mrl(solve_structure(t, hc, rtol=1e-10, atol=1e-12)),
    in_axes=(None, 0)))
_scan = jax.jit(jax.vmap(
    lambda t, hc, n: _mrl(solve_structure(t, hc, backend="scan", n_steps=n)),
    in_axes=(None, 0, None)), static_argnums=2)


@pytest.mark.parametrize("name", list(_EOS))
def test_scan_values_match_diffrax(name):
    # measured over h_c 0.05..0.5 at n_steps=2000: |dM| <= 1.1e-5, |dR| <= 4.4e-6,
    # |dLambda| <= 4.9e-5 (7.8e-5 on bridge1); worst at h_c=0.05 (0.26 Msun).
    # n_steps=8000: <= 4.2e-8, 1.7e-8, 2.0e-7 (6.5e-7): fourth order.
    t = _EOS[name]()
    ref = np.asarray(_ref(t, _HCS))
    for n, tol in ((2000, (3e-5, 1e-5, 1.5e-4)), (8000, (2e-7, 1e-7, 2e-6))):
        err = np.abs(np.asarray(_scan(t, _HCS, n)) / ref - 1.0)
        assert np.all(np.isfinite(err)), (n, err)
        assert np.all(err.max(axis=0) <= np.array(tol)), (n, err)


# ---- reverse-mode gradients ----
_mm = MetaModelEOS()


def _obs(x, **kw):
    pt = solve_structure(_mm.eos_table({"L_sym": x[0]}), x[1], **kw)
    if not kw.get("tidal", True):           # Lambda is a NaN placeholder
        return jnp.stack([pt.M_msun, pt.R_km])
    return _mrl(pt)


_jac_ref = jax.jit(jax.jacfwd(lambda x: _obs(x, rtol=1e-10, atol=1e-12)))
# jax.jacrev: one reverse pass per output row
_jac_scan = jax.jit(jax.jacrev(
    lambda x, tidal: _obs(x, backend="scan", n_steps=2000, tidal=tidal)),
    static_argnums=1)


@pytest.mark.parametrize("hc", [0.1, 0.2, 0.3])
def test_scan_reverse_gradients_match_diffrax(hc):
    # d(M, R, Lambda)/d(L_sym, h_c): jacrev of the scan vs jacfwd of diffrax at
    # rtol 1e-10 (which matches FD at rtol 1e-12 to <= 1.5e-7).  Measured at
    # n_steps=2000: M <= 6.1e-5 (dM/dL_sym at h_c=0.2), R <= 8.0e-6, Lambda <=
    # 2.3e-5; old scan: NaN or O(1).  With a mesh that follows the nodes the M
    # row was 1.2e-2 off; without the EOS-placed points, Lambda up to 9e-3.
    x0 = jnp.array([60.0, hc])
    J = np.asarray(_jac_ref(x0))
    err = np.abs(np.asarray(_jac_scan(x0, True)) / J - 1.0)
    assert np.all(err[:2] <= 2e-4), err        # M, R rows
    assert np.all(err[2] <= 1e-4), err         # Lambda row
    err_s = np.abs(np.asarray(_jac_scan(x0, False)) / J[:2] - 1.0)
    assert np.all(err_s <= 2e-4), err_s        # structure-only


def test_scan_lambda_gradient_keeps_the_node_boundary_terms():
    # The tidal term jumps at every node, and on the steep-joint table of
    # test_tidal_cs2_consistency the nodes above the joint move with the core
    # scale.  Without the scan's node-anchored y picture, dLambda/dscale is off
    # by -1.5 (and dLambda/dh_c by -0.36) at every n_steps; with it, measured
    # 7.1e-6 and 6.2e-6 at n_steps=2000 (M, R <= 1.8e-6) vs FD of diffrax.
    tcs = _sibling("test_tidal_cs2_consistency")

    def obs(x, **kw):
        return _mrl(solve_structure(tcs._steep_table(x[0]), x[1], **kw))

    x0 = jnp.array([1.0, tcs._hc14()])
    val = jax.jit(lambda x: obs(x, rtol=1e-11, atol=1e-13))
    fd = np.stack([(np.asarray(val(x0 + e)) - np.asarray(val(x0 - e))) / (2 * s)
                   for e, s in ((jnp.array([1e-4, 0.0]), 1e-4),
                                (jnp.array([0.0, 1e-5]), 1e-5))], axis=1)
    g = np.asarray(jax.jit(jax.jacrev(
        lambda x: obs(x, backend="scan", n_steps=2000)))(x0))
    err = np.abs(g / fd - 1.0)
    assert np.all(err[:2] <= 2e-5), err
    assert np.all(err[2] <= 5e-5), err


def test_scan_rejects_nonpositive_n_steps():
    # n_steps=0 left the start sigma0 out of the mesh (silently wrong)
    with pytest.raises(ValueError, match="n_steps"):
        solve_structure(_mm.eos_table(), 0.2, backend="scan", n_steps=0)


def test_scan_second_derivatives_are_not_supported():
    """Pins a documented LIMITATION (as test_tov_jump_ts does for diffrax): the
    frozen mesh and the first-order y picture make FIRST derivatives accurate,
    but second derivatives through the scan are O(1) wrong at any n_steps
    (MetaModel h_c = 0.2: d^2(M, R, Lambda)/dh_c^2 off by -0.49, +0.93, -0.096 at
    both n_steps 2000 and 8000; d^2 Lambda/dscale^2 on the steep-joint table
    +7.2).  If this starts failing, update the solve_structure docstring first."""
    t = _mm.eos_table()

    def M(hc):
        return solve_structure(t, hc, backend="scan", n_steps=2000,
                               tidal=False).M_msun

    d2 = float(jax.jit(jax.jacfwd(jax.grad(M)))(0.2))
    g = jax.jit(jax.grad(M))
    fd = (float(g(0.2 + 2e-4)) - float(g(0.2 - 2e-4))) / 4e-4
    assert abs(d2 / fd - 1.0) > 0.1, (d2, fd)
