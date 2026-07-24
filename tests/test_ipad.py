"""Anchor tests for the IPAD-TOV closed forms (pure numpy, fast).

The anchors pin the published GW170817 mapping of 2606.21402 (Sec. III):
X = 0.185 -> Lambda_1.4 ~ 190, X = 0.16 -> ~302, X = 0.133 -> ~580, all at
Psi = 2.66.  They guard the Pi_c = xi/2 convention in Eq. (7): with the
wrong convention the predictions are off by factors of several, not
percent.
"""
import numpy as np
import pytest

from nmbackend.ipad import (
    D_ipad, k2_ipad, lam14_linear, sc2_ipad, xi_ipad,
)

ANCHORS = [
    # (X, Psi, Lambda_expected)  from 2606.21402 Sec. III
    (0.185, 2.66, 190.0),
    (0.160, 2.66, 302.0),
    (0.133, 2.66, 580.0),
]


@pytest.mark.parametrize("X,Psi,lam_ref", ANCHORS)
def test_anchor_lambda(X, Psi, lam_ref):
    lam = lam14_linear(D_ipad(X, Psi))
    assert lam == pytest.approx(lam_ref, rel=0.03)


def test_anchor_sc2_range():
    # their GW170817 inversion: s_c^2 from ~0.62 (X=0.185) to ~0.39 (X=0.133)
    assert sc2_ipad(0.185, 2.66) == pytest.approx(0.62, abs=0.02)
    assert sc2_ipad(0.133, 2.66) == pytest.approx(0.39, abs=0.02)


def test_monotonicity_and_domain():
    X = np.linspace(0.10, 0.30, 40)
    D = D_ipad(X, 2.66)
    ok = np.isfinite(D)
    assert ok.sum() > 30
    assert np.all(np.diff(D[ok]) < 0)          # D decreasing in X
    assert np.all(np.diff(xi_ipad(X, 2.66)) > 0)   # compactness increasing


def test_k2_magnitude():
    # their quote: typical TOV NSs (X >~ 0.15, Psi=0) have 0.02 <~ k2 <~ 0.04
    k2 = k2_ipad(np.array([0.15, 0.20, 0.25]), 0.0)
    assert np.all(k2 > 0.005) and np.all(k2 < 0.08)


def test_broadcasting():
    X = np.array([0.13, 0.16, 0.19])
    out = D_ipad(X, 2.66)
    assert out.shape == X.shape and np.all(np.isfinite(out))
