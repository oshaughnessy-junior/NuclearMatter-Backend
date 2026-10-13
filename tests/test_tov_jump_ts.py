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
from nmbackend.tov.solver import _integrate_diffrax, solve_structure

mm = MetaModelEOS()
_L0 = 60.0
_HC = 0.20 * float(mm.eos_table().h[-1])    # ~1.25 Msun on the default table


def _obs(x, rtol, atol):
    """(M, R, Lambda) at (L_sym, h_c) = x."""
    pt = solve_structure(mm.eos_table({"L_sym": x[0]}), x[1], rtol=rtol, atol=atol)
    return jnp.stack([pt.M_msun, pt.R_km, pt.Lambda])


_val = jax.jit(_obs)
_jac = jax.jit(jax.jacfwd(_obs))             # rows (M, R, Lambda), cols (L_sym, h_c)


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


def _rel_err(rtol, atol):
    global _FD
    if _FD is None:
        _FD = _fd_reference()
    J = np.asarray(_jac(jnp.array([_L0, _HC]), rtol, atol))
    return np.abs(J / _FD - 1.0)


def test_tangents_match_fd_at_default_tolerance():
    # measured <= 1.3e-4 (h_c column) and <= 3e-5 (L_sym column); was 2-10%
    err = _rel_err(1e-7, 1e-9)               # solve_structure's defaults
    assert np.all(err <= 3e-4), err


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
