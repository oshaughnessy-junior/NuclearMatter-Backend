"""Tests for the postmerger hooks (pure numpy, fast)."""
import numpy as np
import pytest

from nmbackend.postmerger import (
    MSUN_S, STROUD2026_EXP, dtilde, fpeak_lnlike, mthresh_lnlike,
)


def test_dtilde_equal_mass_reduces_to_D():
    D = 64.0
    assert dtilde(1.365, 1.365, D, D) == pytest.approx(D, rel=1e-12)


def test_dtilde_symmetry():
    a = dtilde(1.2, 1.5, 40.0, 90.0)
    b = dtilde(1.5, 1.2, 90.0, 40.0)
    assert a == pytest.approx(b, rel=1e-12)


def test_fpk_in_band():
    # production-prior median D-tilde ~ 64 for a GW170817-like binary
    f = STROUD2026_EXP.fpk_hz(2.73, 64.0)
    assert 2000.0 < f < 4000.0


def test_fpk_softer_is_higher():
    # smaller D (softer, more compact) -> higher peak frequency
    f_soft = STROUD2026_EXP.fpk_hz(2.73, 40.0)
    f_stiff = STROUD2026_EXP.fpk_hz(2.73, 115.0)
    assert f_soft > f_stiff


def test_fpeak_lnlike_peaks_at_observation():
    f_pred = np.linspace(2500.0, 3500.0, 101)
    ll = fpeak_lnlike(3000.0, 100.0, f_pred, STROUD2026_EXP.sigma_lnf)
    assert f_pred[np.argmax(ll)] == pytest.approx(3000.0, abs=10.0)


def test_mthresh_guarded():
    with pytest.raises(NotImplementedError):
        mthresh_lnlike()
