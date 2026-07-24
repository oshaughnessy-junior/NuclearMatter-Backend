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


def test_mthresh_anchors_bauswein_table():
    # anchors from the Bauswein 2013 table (verified from the PDF):
    # TMA: Mmax=2.02, R_1.6=13.73 -> Mthres(sim)=3.25
    # IUF: Mmax=1.95, R_1.6=12.57 -> Mthres(sim)=3.05
    from nmbackend.postmerger import BAUSWEIN2013
    assert BAUSWEIN2013.mthresh(2.02, 13.73) == pytest.approx(3.25, abs=0.06)
    assert BAUSWEIN2013.mthresh(1.95, 12.57) == pytest.approx(3.05, abs=0.06)


def test_mthresh_koeppel_lowest_order():
    # their own quote: Mth/M_TOV ~ 1.415 around C_TOV ~ 0.30
    from nmbackend.postmerger import KOEPPEL2019_LIN
    k = KOEPPEL2019_LIN.k(0.30)
    assert k == pytest.approx(1.415, abs=0.02)


def test_mthresh_koeppel_nonlin_lowest_order():
    # Koeppel Eq. (4) passes through their lowest-order quote
    # Mth/M_TOV ~ 1.415 (their Eq. 3) near C_TOV ~ 0.29
    from nmbackend.postmerger import KOEPPEL2019_NONLIN
    assert KOEPPEL2019_NONLIN.k(0.288) == pytest.approx(1.415, abs=0.01)


def test_mthresh_koeppel_nonlin_bh_limit():
    # a = 2b/(2-c) enforces the black-hole limit Mth/M_TOV -> 0 at C = 1/2
    from nmbackend.postmerger import KOEPPEL2019_NONLIN
    assert KOEPPEL2019_NONLIN.k(0.5) == pytest.approx(0.0, abs=1e-9)
    assert KOEPPEL2019_NONLIN.a == pytest.approx(3.0606, abs=1e-3)


def test_mthresh_koeppel_nonlin_monotonic():
    # monotonically decreasing over the neutron-star compactness range
    from nmbackend.postmerger import KOEPPEL2019_NONLIN
    C = np.linspace(0.24, 0.35, 40)
    k = KOEPPEL2019_NONLIN.k(C)
    assert np.all(np.diff(k) < 0)


def test_mthresh_koeppel_nonlin_agrees_with_linear():
    # paper: nonlinear within "8% at most" of the linear variant over the
    # NS range, and within a few percent near C_TOV ~ 0.30
    from nmbackend.postmerger import KOEPPEL2019_NONLIN, KOEPPEL2019_LIN
    C = np.linspace(0.24, 0.35, 12)
    knl = KOEPPEL2019_NONLIN.k(C)
    kl = KOEPPEL2019_LIN.k(C)
    assert np.all(np.abs(knl - kl) / kl < 0.08)
    rel_030 = abs(KOEPPEL2019_NONLIN.k(0.30) - KOEPPEL2019_LIN.k(0.30)) \
        / KOEPPEL2019_LIN.k(0.30)
    assert rel_030 < 0.04


def test_stroud_sigma_lnf_measured():
    # residual refit (Tab. I sim M*f_pk spread + Tab. II R^2 = 0.811);
    # pins the field to the measured ~0.05, not the bare placeholder
    assert STROUD2026_EXP.sigma_lnf == pytest.approx(0.051, abs=0.005)


def test_mthresh_lnlike_behavior():
    # GW170817-like Mtot=2.73, no prompt collapse: favors draws whose
    # threshold is ABOVE Mtot (stiffer/larger Mmax)
    mmax = np.array([1.95, 2.02, 2.30])
    r16 = np.array([10.5, 13.73, 13.5])
    ll = mthresh_lnlike(False, 2.73, 0.02, mmax, r16)
    assert ll.shape == mmax.shape
    assert ll[1] > ll[0]      # TMA-like (Mth~3.2) >> compact-soft draw
    ll_prompt = mthresh_lnlike(True, 2.73, 0.02, mmax, r16)
    assert ll_prompt[0] > ll_prompt[1]
