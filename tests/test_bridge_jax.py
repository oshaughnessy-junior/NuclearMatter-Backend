"""Differentiable bridge (priors.bridge_jax): law, values, and derivatives."""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from nmbackend.priors import bridge as B
from nmbackend.priors import bridge_jax as BJ

NEPS = ["E_sym", "L_sym", "K_sym", "K_sat"]
NEP0 = dict(E_sym=32.8, L_sym=66.8, K_sym=-97.8, K_sat=239.0)
BL = (1042.5, 0.32, 51.3)


def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, x, "right") / a.size
                         - np.searchsorted(b, x, "right") / b.size))


def test_pqcd_anchor_matches_numpy():
    for X in (1.0, 1.7, 2.9, 4.0):
        got = [float(v) for v in BJ.pqcd_anchor(jnp.asarray(math.log(X)))]
        np.testing.assert_allclose(got, B.pqcd_anchor(X), rtol=1e-12)


def test_node_law_matches_numpy_sampler():
    """One node in the root KK volume: same (mu, n, p) law as
    bridge.sample_point_in_volume, though n is placed by one inverse CDF."""
    bH = B.pqcd_anchor(2.0)
    rng = np.random.default_rng(0)
    ref = np.array([B.sample_point_in_volume(rng, BL, bH) for _ in range(4000)])
    U = jnp.asarray(rng.uniform(size=(4000, 3)))
    a = tuple(jnp.asarray(v) for v in BL)
    b = tuple(jnp.asarray(v) for v in bH)
    got = np.asarray(jax.vmap(lambda u: BJ._sample(a, b, u))(U))
    for j in range(3):
        assert _ks(ref[:, j], got[:, j]) < 0.04, j


@pytest.fixture(scope="module")
def draw():
    from nmbackend.eos.metamodel import MetaModelEOS
    model = MetaModelEOS(crust="polytrope")
    th = BJ.draw_theta(np.random.default_rng(3), NEP0, nep_names=NEPS)
    plan = BJ.make_plan(model, th, NEPS)
    return model, th, plan


def test_draw_is_a_valid_eos(draw):
    model, th, plan = draw
    n, eps, p, info = BJ.bridge_arrays(model, jnp.asarray(th), NEPS, plan)
    n, eps, p = map(np.asarray, (n, eps, p))
    assert n.size == plan.n_keep + plan.n_sel
    assert np.all(np.diff(n) > 0) and np.all(np.diff(eps) > 0) and np.all(np.diff(p) >= 0)
    mu = np.asarray(info["mu"])
    cs2 = np.diff(np.log(mu)) / np.diff(np.log(np.asarray(info["grid"])))
    assert cs2.min() >= -1e-12 and cs2.max() <= 1.0 + 1e-9
    assert abs(float(info["p_end_mismatch"])) < 0.2
    pts = np.asarray(info["pts"])
    assert np.all(np.diff(pts[:, 0]) >= 0) and np.all(np.diff(pts[:, 1]) >= 0)


def test_gradient_matches_central_differences(draw):
    """d ln p(node) / d theta for the nuclear parameters, corr_frac, ln X and a
    few variates, against central differences at the same variates."""
    model, th, plan = draw
    j = plan.n_keep + 100
    f = lambda t: jnp.log(BJ.bridge_arrays(model, t, NEPS, plan)[2][j])
    g = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(th)))
    assert np.all(np.isfinite(g))
    fv = jax.jit(f)
    for i in (0, 1, 2, 3, 4, 5, 6, 10, 30):
        h = 1e-5 * max(abs(th[i]), 1.0)
        e = np.zeros(th.size); e[i] = h
        fd = (float(fv(jnp.asarray(th + e))) - float(fv(jnp.asarray(th - e)))) / (2 * h)
        assert fd == pytest.approx(g[i], rel=1e-5, abs=1e-10), i


def test_gradient_endpoints_of_ln_X():
    """At ln X = 0 and ln 4 the derivative is the one-sided one from inside;
    outside it is zero (ln X clamped)."""
    for x in (0.0, math.log(4.0)):
        for j, c in ((1, BJ._C_N), (2, BJ._C_P)):
            g = float(jax.grad(lambda t: BJ.pqcd_anchor(t)[j])(x))
            assert g == pytest.approx(np.polyval(np.polyder(c), x), rel=1e-12)
    for x in (-0.1, math.log(4.0) + 0.1):
        assert float(jax.grad(lambda t: BJ.pqcd_anchor(t)[2])(x)) == 0.0


# ------------------------------------------------ failures are visible --------
class _Tab(NamedTuple):
    n: jnp.ndarray
    eps_mevfm3: jnp.ndarray
    p_mevfm3: jnp.ndarray


class _StubModel:
    """Low-density table p = A n^2, eps = 939 n + p on n = s * [0.05, 2]."""

    def eos_table(self, nep):
        n = nep["s"] * jnp.geomspace(0.05, 2.0, 40)
        p = nep["A"] * n ** 2
        return _Tab(n, 939.0 * n + p, p)


def test_invalid_draws_are_flagged():
    names = ["A", "s"]
    cfg = BJ.Config(levels=4)
    z = np.zeros((2 ** cfg.levels - 1) * 3)
    th = lambda A, s=1.0: np.r_[A, s, 0.3, 0.5, z]  # noqa: E731
    model = _StubModel()
    with pytest.raises(ValueError, match="anchors"):
        BJ.make_plan(model, th(1e5), names, cfg)
    with pytest.raises(ValueError, match="must contain"):
        BJ.make_plan(model, th(100.0), names, cfg._replace(n_L_fac=100.0))
    plan = BJ.make_plan(model, th(100.0), names, cfg)
    with pytest.raises(ValueError, match="other settings"):
        BJ.bridge_arrays(model, jnp.asarray(th(100.0)), names, plan,
                         cfg._replace(n_term_fac=math.inf))
    with pytest.raises(ValueError, match="anchors consistent False"):
        BJ.bridge_arrays(model, jnp.asarray(th(1e5)), names, plan, cfg)
    f = jax.jit(lambda t: BJ.bridge_arrays(model, t, names, plan, cfg))
    _, _, p, info = f(jnp.asarray(th(100.0)))
    assert bool(info["valid"]) and np.all(np.isfinite(np.asarray(p)))
    # numpy raises here (anchors inconsistent); under jit: flag and NaN
    _, _, p, info = f(jnp.asarray(th(1e5)))
    assert not bool(info["valid"]) and not bool(info["anchors_ok"])
    assert np.all(np.isnan(np.asarray(p)))
    # a low-density grid that moved: plan.n_keep no longer fits
    _, _, p, info = f(jnp.asarray(th(100.0, 1.3)))
    assert not bool(info["plan_ok"]) and np.all(np.isnan(np.asarray(p)))
    # n_L outside the low-density table
    _, _, p, info = f(jnp.asarray(th(100.0, 10.0)))
    assert not bool(info["in_table"]) and np.all(np.isnan(np.asarray(p)))
    # the flag does not reach the derivatives of a valid draw
    g = jax.jit(jax.grad(lambda t: f(t)[2][plan.n_keep + 50]))(jnp.asarray(th(100.0)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_fixed_mass_ln_lambda_gradient(draw):
    """d ln Lambda / d theta at fixed mass through the TOV solve (implicit
    function rule on h_c) against central differences that re-solve h_c."""
    from nmbackend.tov.solver import solve_structure
    model, th, plan = draw
    idx = np.array([1, 4, 5, 30])           # L_sym, corr_frac, ln X, a variate
    th0 = jnp.asarray(th)

    def ml(q):                              # q = (theta[idx], h_c)
        s = solve_structure(BJ.bridge_table(model, th0.at[idx].set(q[:-1]), NEPS, plan), q[-1])
        return jnp.stack([s.M_msun, jnp.log(s.Lambda)])

    f, J = jax.jit(ml), jax.jit(jax.jacfwd(ml))
    h_top = float(np.asarray(jax.jit(lambda t: BJ.bridge_table(model, t, NEPS, plan).h)(th0))[-1])
    q0 = np.r_[th[idx], 0.3 * h_top]
    M0 = float(f(q0)[0])
    Jq = np.asarray(J(q0))
    ad = Jq[1, :-1] - Jq[1, -1] * Jq[0, :-1] / Jq[0, -1]

    def lnlam_at_M0(q):
        q = q.copy()
        for _ in range(5):
            q[-1] -= (float(f(q)[0]) - M0) / Jq[0, -1]
        v = f(q)
        assert abs(float(v[0]) - M0) < 1e-11
        return float(v[1])

    for j in range(idx.size):
        h = 1e-5 * max(abs(q0[j]), 1.0)
        e = np.zeros(q0.size)
        e[j] = h
        fd = (lnlam_at_M0(q0 + e) - lnlam_at_M0(q0 - e)) / (2 * h)
        assert ad[j] == pytest.approx(fd, rel=1e-4), idx[j]
