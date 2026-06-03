"""Validate the constrained Brownian-bridge prior (GKKS arXiv:2512.18044).

Every draw must be mechanically stable (mu, p monotone), causal (cs^2 in [0,1] --
exact via the maximum principle of the (ln mu, ln n) diffusion), thermodynamically
consistent (d eps/dn = mu), anchored at both ends, and TOV-viable.
"""
import numpy as np
import pytest

from nmbackend.priors import (kk_volume, sample_point_in_volume, fractal_refine,
                              pqcd_anchor, draw_bridge_eos)

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


def test_draw_makes_a_neutron_star():
    from nmbackend.tov import build_family, m_max
    from nmbackend.tov.family import default_hc_grid
    d = draw_bridge_eos(np.random.default_rng(11),
                        nep_params={"L_sym": 60.0}, corr_frac=0.3)
    t = d["table"]
    fam = build_family(t, default_hc_grid(t, 80), rtol=1e-7, atol=1e-9)
    mm = float(m_max(fam))
    assert np.isfinite(mm) and 1.0 < mm < 3.5
