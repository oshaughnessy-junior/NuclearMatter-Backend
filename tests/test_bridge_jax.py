"""Differentiable bridge (priors.bridge_jax): law, values, and derivatives."""
import math

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
