"""Forward-mode tangents of the adaptive TOV solve match finite differences.

The TOV right-hand side has slope kinks at every EOS table node (eps from the
C^1 Hermite of p(h), cs2 linear).  Without ``jump_ts`` at the nodes, Dopri5 steps
straddled them (~half of all steps rejected) and ForwardMode differentiated the
mesh's position relative to the kinks: on this table the partials were off by
up to 10% at the default rtol, 1% at 1e-9 and 4% at 1e-11, erratic in rtol
(docs/devlog/2026-10-12-tov-jump-ts.md).  All four checks fail on that solver.
"""
import jax
import jax.numpy as jnp
import numpy as np

from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family
from nmbackend.tov.solver import _integrate_diffrax, solve_structure

mm = MetaModelEOS()
_L0 = 60.0
_HC = 0.20 * float(mm.eos_table().h[-1])    # ~1.25 Msun on the default table


def _obs(x, rtol, atol, centre_refine=False):
    """(M, R, Lambda) at (L_sym, h_c) = x."""
    pt = solve_structure(mm.eos_table({"L_sym": x[0]}), x[1], rtol=rtol, atol=atol,
                         centre_refine=centre_refine)
    return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])


_val = jax.jit(_obs)
# rows (M, R, Lambda), cols (L_sym, h_c)
_jac = jax.jit(jax.jacfwd(_obs), static_argnums=3)


def _fd_reference():
    x0 = np.array([_L0, _HC])
    steps = np.array([0.02, 1e-4 * _HC])
    cols = []
    for j in range(2):
        e = np.zeros(2)
        e[j] = steps[j]
        cols.append((_val(x0 + e, 1e-11, 1e-13) - _val(x0 - e, 1e-11, 1e-13))
                    / (2.0 * steps[j]))
    return np.stack(cols, axis=1)


_FD = None


def _rel_err(rtol, atol, centre_refine=False):
    global _FD
    if _FD is None:
        _FD = _fd_reference()
    J = np.asarray(_jac(jnp.array([_L0, _HC]), rtol, atol, centre_refine))
    return np.abs(J / _FD - 1.0)


def test_tangents_match_fd_at_default_tolerance():
    # measured <= 1.2e-4 (h_c column: the frozen start) and <= 3e-5 (L_sym
    # column: jump_ts); was up to 10%.  The h_c residual is a near-common factor
    # (+1.0e-4, +1.2e-4, +1.0e-4 on M, R, Lambda), so it largely cancels in
    # fixed-mass (IFT) derivatives.
    err = _rel_err(1e-7, 1e-9)               # solve_structure's defaults
    assert np.all(err <= 3e-4), err


def test_centre_refine_tightens_the_hc_column():
    # opt-in: inner 5% of the star at rtol*1e-3; measured h_c column <= 8.1e-6,
    # L_sym column unchanged (<= 3.0e-5)
    err = _rel_err(1e-7, 1e-9, True)
    assert np.all(err[:, 1] <= 3e-5), err
    assert np.all(err[:, 0] <= 1e-4), err


def test_tangents_match_fd_at_tight_tolerance():
    # measured <= 8e-7; was up to 2% at rtol 1e-9
    err = _rel_err(1e-9, 1e-11)
    assert np.all(err <= 1e-5), err


def test_steps_stop_at_nodes_instead_of_rejecting():
    # unpatched: 165 of 391 steps rejected at the default rtol
    t = mm.eos_table({"L_sym": _L0})
    st = jax.jit(lambda h: _integrate_diffrax(t, h, 1e-7, 1e-9, True).stats)(_HC)
    assert int(st["num_rejected_steps"]) < 0.1 * int(st["num_accepted_steps"])


def test_batched_tables_match_unbatched():
    # jump_ts is a per-table array; under vmap over EOS parameters it is batched.
    # Patched solver: equal to 1e-12 (unpatched: 2e-5, as roundoff flipped
    # accept/reject decisions at the kinks).  1e-10 leaves roundoff headroom.
    xs = jnp.array([[55.0, _HC], [65.0, 1.1 * _HC]])
    batched = np.asarray(jax.vmap(_val, in_axes=(0, None, None))(xs, 1e-7, 1e-9))
    single = np.stack([np.asarray(_val(x, 1e-7, 1e-9)) for x in xs])
    np.testing.assert_allclose(batched, single, rtol=1e-10, atol=0.0)


def test_second_hc_derivative_is_not_supported():
    """Pins a documented LIMITATION: the frozen start makes first h_c derivatives
    (and mixed d^2/dh_c dtheta: measured 9.3e-4 at the default rtol, 2.4e-7 at
    1e-10) accurate, but d^2 X/dh_c^2 is wrong by orders of magnitude (AD/FD - 1
    = +3.7e4 at the default rtol, +7.9 at 1e-10; the base solver gave O(1)).  If
    this starts failing, second h_c derivatives may have become usable: update
    the solve_structure docstring, README and docs/physics.rst first."""
    t = mm.eos_table({"L_sym": _L0})

    def M(hc, rtol):
        return solve_structure(t, hc, tidal=False, rtol=rtol,
                               atol=rtol * 1e-2).M_msun

    grad = jax.jit(jax.jacfwd(M))
    d2 = float(jax.jit(jax.jacfwd(jax.jacfwd(M)))(_HC, 1e-7))
    dh = 1e-4 * _HC
    fd = (float(grad(_HC + dh, 1e-10)) - float(grad(_HC - dh, 1e-10))) / (2 * dh)
    assert abs(d2 / fd - 1.0) > 10.0


def test_jump_nodes_opt_out_is_finite_and_close():
    # value-only opt-out (not the pre-2026-10-12 solver; see solve_structure).
    # Values within 1e-3 of the default (measured 5.4e-6 in Lambda here; up to
    # 7.7e-4 near M_max on this table), finite
    # tangents, and reachable through build_family.
    t = mm.eos_table({"L_sym": _L0})

    def f(hc, jn):
        pt = solve_structure(t, hc, jump_nodes=jn)
        return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])

    on, off = (np.asarray(jax.jit(f, static_argnums=1)(_HC, jn)) for jn in (True, False))
    np.testing.assert_allclose(off, on, rtol=1e-3, atol=0.0)
    tan = jax.jit(lambda h: jax.jvp(lambda x: f(x, False), (h,), (1.0,))[1])(_HC)
    assert np.all(np.isfinite(np.asarray(tan)))
    hcs = jnp.array([0.9, 1.0, 1.1]) * _HC
    fam_off = build_family(t, hcs, jump_nodes=False)
    fam_on = build_family(t, hcs)
    # batched vs unbatched agree only to roundoff-flipped step decisions here
    # (measured 3.0e-5 in Lambda, 2.3e-8 in M), and the flag must reach the solve
    np.testing.assert_allclose(np.asarray(fam_off.Lambda)[1], off[2], rtol=2e-4,
                               atol=0.0)
    assert np.any(np.asarray(fam_off.Lambda) != np.asarray(fam_on.Lambda))
