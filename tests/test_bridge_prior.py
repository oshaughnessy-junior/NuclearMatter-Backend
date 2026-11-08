"""Validate the constrained Brownian-bridge prior (GKKS arXiv:2512.18044).

Every draw must be mechanically stable (mu, p monotone), causal (cs^2 in [0,1] --
exact via the maximum principle of the (ln mu, ln n) diffusion), thermodynamically
consistent (d eps/dn = mu), anchored at both ends, and TOV-viable.
"""
import numpy as np
import pytest

from nmbackend.eos.metamodel import DEFAULT_PARAMS, MetaModelEOS
from nmbackend.priors import (kk_volume, sample_point_in_volume, fractal_refine,
                              pqcd_anchor, draw_bridge_eos,
                              draw_prior_ensemble)
from nmbackend.priors.bridge import _table_pts_bound

# KK 2022 Table I "stiff" CET anchor (mu MeV, n fm^-3, p MeV/fm^3)
_BL = (978.0, 0.176, 3.542)
_BH = pqcd_anchor(2.0)


def test_pqcd_anchor_matches_table():
    # exact at the tabulated X values (KK 2022 Table I / Gorda et al. 2021)
    for X, (n_H, p_H) in {1.0: (6.14, 2334.0), 2.0: (6.47, 3823.0),
                          4.0: (6.87, 4284.0)}.items():
        mu, n, p = pqcd_anchor(X)
        assert mu == 2600.0
        assert abs(n - n_H) < 1e-6 and abs(p - p_H) < 1e-6


def test_uniform_draws_inside_volume():
    n_min, n_max, p_min, p_max, mu_c = kk_volume(_BL, _BH)
    assert _BL[0] < mu_c < _BH[0]
    rng = np.random.default_rng(1)
    for _ in range(500):
        mu, n, p = sample_point_in_volume(rng, _BL, _BH)
        assert n_min(mu) - 1e-9 <= n <= n_max(mu) + 1e-9
        assert p_min(mu) - 1e-6 <= p <= p_max(mu, n) + 1e-6


def test_fractal_is_monotone_and_causal_chords():
    rng = np.random.default_rng(2)
    pts = fractal_refine(rng, _BL, _BH, levels=8)
    mu = np.array([q[0] for q in pts])
    n = np.array([q[1] for q in pts])
    p = np.array([q[2] for q in pts])
    assert np.all(np.diff(mu) > 0) and np.all(np.diff(n) >= -1e-12)
    assert np.all(np.diff(p) >= -1e-9)
    cs2 = np.diff(np.log(mu)) / np.maximum(np.diff(np.log(n)), 1e-300)
    assert cs2.min() >= -1e-12 and cs2.max() <= 1.0 + 1e-9


def test_draw_is_causal_stable_consistent_and_anchored():
    rng = np.random.default_rng(3)
    for _ in range(3):
        d = draw_bridge_eos(rng, corr_frac=float(rng.uniform(0.2, 0.4)))
        br, m = d["bridge"], d["meta"]
        # causality (exact max principle) and stability
        assert br["cs2"].min() >= -1e-10 and br["cs2"].max() <= 1.0 + 1e-9
        assert np.all(np.diff(br["mu"]) >= 0)
        assert np.all(np.diff(br["p"]) >= -1e-9)
        # thermodynamic consistency d eps/dn = mu (interior points)
        mu_num = np.gradient(br["eps"], br["n"])[2:-2]
        rel = np.abs(mu_num - br["mu"][2:-2]) / br["mu"][2:-2]
        assert np.median(rel) < 1e-3
        # anchors: mu pinned exactly (Dirichlet); pQCD pressure within the X band
        assert abs(br["mu"][-1] - 2600.0) < 1e-6
        assert abs(m["p_end_mismatch"]) < 0.10


@pytest.mark.parametrize("crust", ["sly", "polytrope"])
def test_anchor_is_the_model_at_n_L(crust):
    """The core grid ends at n_match = n_L, so its last node is the model at
    n_L, before any crust or table step.  The anchor used to come from the node
    below it: p_L low by 1.5% (SLy) and 6.7% (polytrope)."""
    model = MetaModelEOS(crust=crust)
    d = draw_bridge_eos(np.random.default_rng(4), model=model, corr_frac=0.3)
    m = d["meta"]
    n_m, eps_m, p_m = (float(a[-1]) for a in model._low_density(DEFAULT_PARAMS))
    assert n_m == pytest.approx(m["n_L"], rel=1e-14)
    assert m["p_L"] == pytest.approx(p_m, rel=1e-12)
    assert m["mu_L"] * m["n_L"] - m["p_L"] == pytest.approx(eps_m, rel=1e-12)

    # the join: one node at n_L, on the model, and p and eps rise through it
    n, eps, p = d["n"], d["eps"], d["p"]
    at_n_L = np.isclose(n, m["n_L"], rtol=1e-9, atol=0.0)
    assert at_n_L.sum() == 1
    assert p[at_n_L][0] == pytest.approx(p_m, rel=1e-12)
    assert eps[at_n_L][0] == pytest.approx(eps_m, rel=1e-12)
    assert np.diff(np.log(n)).min() > 1e-3
    assert np.all(np.diff(eps) > 0) and np.all(np.diff(p) > 0)

    bound = _table_pts_bound(model.eos_table().n.shape[0], m["n_L"],
                             12.0 * 0.16, 600)
    assert n.size <= bound and d["table"].n.shape[0] == bound


def _low_table():
    t = MetaModelEOS().eos_table()
    return dict(n=np.array(t.n), eps=np.array(t.eps_mevfm3),
                p=np.array(t.p_mevfm3))


def test_anchor_between_table_nodes():
    """n_L = 1.5 n_sat is not a table node; a core grid ending there gives the
    model value."""
    d = draw_bridge_eos(np.random.default_rng(4), corr_frac=0.3, n_L_fac=1.5)
    m = d["meta"]
    _, eps_m, p_m = (float(a[-1]) for a in MetaModelEOS()._low_density(
        dict(DEFAULT_PARAMS, n_match_fac=1.5)))
    assert m["p_L"] == pytest.approx(p_m, rel=1e-3)
    assert m["mu_L"] * m["n_L"] - m["p_L"] == pytest.approx(eps_m, rel=1e-4)


def test_table_node_within_roundoff_of_n_L_is_not_a_second_node():
    low = _low_table()
    low["n"][np.argmin(np.abs(low["n"] - 0.32))] = 0.32 * (1.0 - 1e-14)
    d = draw_bridge_eos(np.random.default_rng(4), corr_frac=0.3, low_table=low)
    assert np.diff(np.log(d["n"])).min() > 1e-3


@pytest.mark.parametrize("nodes", [lambda n: n < 0.3, lambda n: n > 0.4,
                                   lambda n: n >= 0.32],
                         ids=["ends_below", "starts_above", "starts_at"])
def test_low_table_must_contain_n_L(nodes):
    low = _low_table()
    part = {k: v[nodes(low["n"])] for k, v in low.items()}
    with pytest.raises(ValueError, match="must contain"):
        draw_bridge_eos(np.random.default_rng(4), low_table=part)


def test_ensemble_warns_when_short_of_draws():
    low = _low_table()
    short = {k: v[low["n"] < 0.3] for k, v in low.items()}
    with pytest.warns(RuntimeWarning, match="0 of 2 draws"):
        tables, _ = draw_prior_ensemble(2, low_table=short)
    assert tables == []


def test_draw_makes_a_neutron_star():
    from nmbackend.tov import build_family, m_max
    from nmbackend.tov.family import default_hc_grid
    d = draw_bridge_eos(np.random.default_rng(11),
                        nep_params={"L_sym": 60.0}, corr_frac=0.3)
    t = d["table"]
    fam = build_family(t, default_hc_grid(t, 80), rtol=1e-7, atol=1e-9)
    mm = float(m_max(fam))
    assert np.isfinite(mm) and 1.0 < mm < 3.5


def test_default_grid_starts_on_the_stable_branch():
    """The minimum-mass star of the MetaModel crust sits at n_c = 0.093
    fm^-3, above the old n_c = 0.07 grid start, whose first star had
    M = 0.62, R = 1840 km (a true TOV solution, below the minimum mass)."""
    from nmbackend.tov import build_family
    from nmbackend.tov.family import default_hc_grid
    t = draw_bridge_eos(np.random.default_rng(1), corr_frac=0.3)["table"]
    fam = build_family(t, default_hc_grid(t, 40))
    M = np.asarray(fam.M_msun)
    top = int(np.argmax(np.where(np.asarray(fam.stable), M, -np.inf)))
    assert np.all(np.diff(M[:top + 1]) > 0)
    assert 0.08 < M[0] < 0.15 and float(fam.R_km[0]) < 100.0
