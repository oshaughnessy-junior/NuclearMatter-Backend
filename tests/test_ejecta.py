"""Tests for the ejecta-conditions surrogate (bridge "mmapipe" segment).

Anchor values marked "paper-quoted" come from the source PDFs themselves
(house rule); anchors marked "pinned" are self-computed at implementation
time after PDF-verifying the coefficients, and guard against transcription
drift only.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.constants import GMSUN_KM
from nmbackend.ejecta import (
    EjectaInputs, MEJ_FAMILIES, baryonic_mass, ejecta_conditions,
    ejecta_from_family, kf2020_mdisk, kf2020_mdyn, lambda_tilde,
    lanthanide_weight, nedora2021_mej, nedora2021_ye,
    prompt_collapse_disk_factor, radice2018_mdisk, radice2018_mej,
    radice2018_vej,
)

# A GW170817-like fiducial: 1.35+1.35, R = 11.9 km -> C ~ 0.1675, Lambda ~ 400.
_FID = dict(m1=1.35, m2=1.35, lambda1=400.0, lambda2=400.0,
            r1_km=11.9, r2_km=11.9)


def _inputs(**kw):
    d = {**_FID, **kw}
    return EjectaInputs(*[jnp.asarray(d[k], float) for k in
                          ("m1", "m2", "lambda1", "lambda2", "r1_km", "r2_km")])


# ---------------------------------------------------------------- helpers ---

def test_lambda_tilde_equal_mass_reduces_to_lambda():
    assert float(lambda_tilde(1.4, 1.4, 300.0, 300.0)) == pytest.approx(300.0, rel=1e-12)


def test_lambda_tilde_symmetry():
    a = float(lambda_tilde(1.2, 1.5, 800.0, 200.0))
    b = float(lambda_tilde(1.5, 1.2, 200.0, 800.0))
    assert a == pytest.approx(b, rel=1e-12)


def test_baryonic_mass_exceeds_gravitational():
    # KF2020 Eq. (7): binding raises M_b above M; ~10% at C ~ 0.17
    mb = float(baryonic_mass(1.35, 0.167))
    assert 1.35 < mb < 1.55
    assert mb == pytest.approx(1.35 * (1 + 0.6 * 0.167 / (1 - 0.5 * 0.167)), rel=1e-12)


# ----------------------------------------------------------- fit families ---

def test_kf2020_mdisk_paper_anchor():
    # paper-quoted (KF20 Table I): C1=0.167, M1=1.5 -> M_disk^KF = 0.037 Msun
    r = GMSUN_KM * 1.5 / 0.167
    x = _inputs(m1=1.5, m2=1.5, r1_km=r, r2_km=r)
    assert float(kf2020_mdisk(x)) == pytest.approx(0.037, abs=0.001)


def test_kf2020_mdyn_vanishes_at_extremes():
    # KF20 Sec. III: no dynamical ejecta for very compact (prompt collapse)
    # or very extended stars; positive in between.
    def mdyn_at(c):
        r = GMSUN_KM * 1.35 / c
        return float(kf2020_mdyn(_inputs(r1_km=r, r2_km=r)))
    assert mdyn_at(0.10) == 0.0
    assert mdyn_at(0.25) == 0.0
    assert mdyn_at(0.166) > 1.0e-3


def test_radice2018_mej_pinned():
    # pinned (coefficients PDF-verified): 1.35+1.35, C=0.167 -> ~4.8e-4 Msun,
    # inside the simulated dynamical-ejecta range O(1e-4 - 1e-2).
    v = float(radice2018_mej(_inputs()))
    assert v == pytest.approx(4.8e-4, rel=0.15)


def test_radice2018_vej_band():
    # Radice+18 Fig. 6: fiducial ejecta velocities ~ 0.1 - 0.3 c
    v = float(radice2018_vej(_inputs()))
    assert 0.1 < v < 0.3


def test_radice2018_mdisk_limits():
    # Eq. (25): floor at 1e-3 for very compact binaries; alpha+beta ~ 0.211
    # in the stiff limit; monotonically increasing in Lambda-tilde.
    lo = float(radice2018_mdisk(_inputs(lambda1=50.0, lambda2=50.0)))
    hi = float(radice2018_mdisk(_inputs(lambda1=3000.0, lambda2=3000.0)))
    assert lo == pytest.approx(1.0e-3, abs=1e-12)
    assert hi == pytest.approx(0.084 + 0.127, abs=0.01)


def test_nedora2021_ye_paper_anchor():
    # paper-quoted: restricted common range q in (1,1.2), Lt in (350,850)
    # has mean <Ye> = 0.20 +/- 0.04 (M0RefSet+M0/M1Set)
    x = _inputs(m1=1.45, m2=1.32, lambda1=500.0, lambda2=800.0)  # q~1.1, Lt~630
    assert float(nedora2021_ye(x)) == pytest.approx(0.20, abs=0.04)


def test_asymmetric_pinned_anchors():
    # pinned (self-computed after PDF verification; jax 0.10.1 float64):
    # closes the equal-mass blind spot -- at q=1 the mass-ratio exponents
    # (Radice n=-0.773, KF20 n=1.5465) and all Nedora q-coefficients are
    # invisible to the other anchors, so a wrong sign/row there would pass.
    # The anchor is given by compactness C = GMSUN_KM m / R (what the fits use),
    # so the pins do not depend on the unit constant.  C1, C2 are the
    # compactnesses the pins were computed at: R = 11.5, 12.0 km with the
    # pre-2026-10-14 GMSUN_KM = 1.4770039 km.
    from nmbackend.ejecta import (nedora2021_vej, nedora2021_mdisk)
    C1, C2 = 0.19265267846681078, 0.1415462040401985
    x = _inputs(m1=1.50, m2=1.15, lambda1=250.0, lambda2=900.0,
                r1_km=GMSUN_KM * 1.50 / C1,
                r2_km=GMSUN_KM * 1.15 / C2)     # q=1.3043, Lt=466.26
    assert float(radice2018_mej(x)) == pytest.approx(8.97442575e-04, rel=1e-6)
    assert float(radice2018_vej(x)) == pytest.approx(2.09410824e-01, rel=1e-6)
    assert float(kf2020_mdyn(x)) == pytest.approx(1.05719335e-02, rel=1e-6)
    assert float(kf2020_mdisk(x)) == pytest.approx(1.60880132e-01, rel=1e-6)
    assert float(nedora2021_mej(x)) == pytest.approx(3.68737158e-03, rel=1e-6)
    assert float(nedora2021_vej(x)) == pytest.approx(1.79986160e-01, rel=1e-6)
    assert float(nedora2021_ye(x)) == pytest.approx(1.80874554e-01, rel=1e-6)
    assert float(nedora2021_mdisk(x)) == pytest.approx(1.79224185e-01, rel=1e-6)
    assert float(radice2018_mdisk(x)) == pytest.approx(5.30252607e-02, rel=1e-6)


def test_physical_ceilings_under_extrapolation():
    # OUR regularizations (documented in the docstrings): velocity respects
    # the kinematic bound where the Nedora polynomial turns superluminal,
    # and disk masses cannot exceed the lighter star.
    from nmbackend.ejecta import nedora2021_vej, nedora2021_mdisk
    x_stiff = _inputs(m1=1.4, m2=1.4, lambda1=3000.0, lambda2=3000.0)
    assert float(nedora2021_vej(x_stiff)) <= 0.99
    x_big = _inputs(m1=1.1, m2=1.1, lambda1=2500.0, lambda2=2500.0,
                    r1_km=13.5, r2_km=13.5)
    assert float(nedora2021_mdisk(x_big)) <= 1.1
    assert float(kf2020_mdisk(_inputs(r1_km=25.0, r2_km=25.0))) <= 1.35


def test_equal_mass_unequal_radius_tiebreak_symmetric():
    # EOS-inconsistent but accepted input: the light() tie-break on
    # compactness must make the disk invariant under label swap.
    a = kf2020_mdisk(_inputs(m1=1.35, m2=1.35, r1_km=11.0, r2_km=12.5))
    b = kf2020_mdisk(_inputs(m1=1.35, m2=1.35, r1_km=12.5, r2_km=11.0))
    assert float(a) == pytest.approx(float(b), rel=1e-12)


def test_single_family_registry_grad_finite():
    # sigma_formula is identically 0 for a one-family registry; its gradient
    # must be 0, not NaN (where-guarded sqrt in _combine).
    reg = {"radice2018": MEJ_FAMILIES["radice2018"]}
    g = jax.grad(lambda m: ejecta_conditions(
        m, 1.35, 400.0, 400.0, 11.9, 11.9, mej_families=reg).m_dyn.sigma)(1.35)
    assert np.isfinite(float(g))


def test_nedora2021_mej_band():
    # paper-quoted band: dynamical ejecta O(1e-4 - 1e-2) Msun
    assert 1.0e-4 < float(nedora2021_mej(_inputs())) < 1.0e-2


# ------------------------------------------------------ combined surrogate ---

def test_conditions_carry_uncertainties():
    ec = ejecta_conditions(**_FID)
    for uq in (ec.m_dyn, ec.v_dyn, ec.m_disk, ec.ye_dyn, ec.x_lanthanide):
        assert np.isfinite(float(uq.value))
        assert float(uq.sigma_fit) > 0.0
        assert float(uq.sigma) >= float(uq.sigma_fit)
    # several families disagree -> nonzero inter-formula systematic
    assert float(ec.m_dyn.sigma_formula) > 0.0
    assert float(ec.m_disk.sigma_formula) > 0.0
    assert set(ec.m_dyn_by) == set(MEJ_FAMILIES)


def test_conditions_physical_ranges():
    ec = ejecta_conditions(**_FID)
    assert 1.0e-5 < float(ec.m_dyn.value) < 0.05
    assert 0.05 < float(ec.v_dyn.value) < 0.5
    assert 1.0e-4 < float(ec.m_disk.value) < 0.5
    assert 0.01 < float(ec.ye_dyn.value) < 0.5
    assert 0.0 <= float(ec.f_tidal.value) <= 1.0
    assert 0.0 <= float(ec.x_lanthanide.value) <= 1.0


def test_mixture_inversion_consistent():
    # f_tidal reproduces <Ye> when mixed back through the two components
    ec = ejecta_conditions(**_FID)
    f = float(ec.f_tidal.value)
    ye = f * float(ec.ye_tidal.value) + (1 - f) * float(ec.ye_shocked.value)
    assert ye == pytest.approx(float(ec.ye_dyn.value), abs=1e-6)


def test_lanthanide_weight_step():
    assert float(lanthanide_weight(0.05)) > 0.95
    assert float(lanthanide_weight(0.45)) < 0.05


def test_prompt_collapse_gate():
    # KF20: M_tot >= 0.95 M_thr -> no significant disk (tanh width 0.06)
    assert float(prompt_collapse_disk_factor(3.75, 3.0)) < 0.01
    assert float(prompt_collapse_disk_factor(2.0, 3.0)) > 0.99
    ec_gated = ejecta_conditions(**_FID, m_thresh=2.30)   # Mtot/Mthr ~ 1.17
    ec_free = ejecta_conditions(**_FID)
    assert float(ec_gated.m_disk.value) < 0.1 * float(ec_free.m_disk.value)
    # deep in collapse the gated sigma stays at least the published disk
    # floors -- the papers assert "no significant disk", not a precise zero
    assert float(ec_gated.m_disk.sigma_fit) >= 5.0e-4


def test_spin_accepted_and_inert():
    # forward-compatibility: spins are carried but current families ignore them
    a = ejecta_conditions(**_FID)
    b = ejecta_conditions(**_FID, chi1=0.3, chi2=-0.1)
    assert float(a.m_dyn.value) == pytest.approx(float(b.m_dyn.value), rel=1e-12)


def test_family_registry_extension():
    # a user-supplied (e.g. spin-aware) family enters through the registry
    from nmbackend.ejecta import FitFamily
    reg = dict(MEJ_FAMILIES)
    reg["toy_spin"] = FitFamily(
        "toy_spin", "none", lambda x: 1.0e-3 * (1.0 + x.chi1 + x.chi2),
        lambda v, x: 0.5 * v, "test stub")
    ec = ejecta_conditions(**_FID, chi1=0.2, mej_families=reg)
    assert "toy_spin" in ec.m_dyn_by
    assert float(ec.m_dyn_by["toy_spin"]) == pytest.approx(1.2e-3, rel=1e-12)
    # and the spin gradient now flows
    g = jax.grad(lambda c: ejecta_conditions(
        **_FID, chi1=c, mej_families=reg).m_dyn.value)(0.2)
    assert float(g) == pytest.approx(1.0e-3 / len(reg), rel=1e-6)


# --------------------------------------------------------------- gradients ---

def test_grad_wrt_binary_and_eos_inputs():
    def f(m1, l1, r1):
        ec = ejecta_conditions(m1, 1.35, l1, 400.0, r1, 11.9)
        return ec.m_dyn.value + ec.m_disk.value + ec.ye_dyn.value
    g = jax.grad(f, argnums=(0, 1, 2))(1.35, 400.0, 11.9)
    assert all(np.isfinite(float(gi)) for gi in g)
    # radius must matter (compactness-driven families)
    assert abs(float(g[2])) > 0.0


def test_grad_sigma_also_differentiable():
    # the error bars themselves are part of the differentiable output
    g = jax.grad(lambda m: ejecta_conditions(
        m, 1.35, 400.0, 400.0, 11.9, 11.9).m_dyn.sigma)(1.35)
    assert np.isfinite(float(g))


# ------------------------------------------- end-to-end bridge composition ---

def test_dmej_dLsym_through_metamodel_matches_fd():
    """d M_ej / d L_sym through metamodel + TOV + ejecta vs finite differences.

    This is the bridge contract: gradients with respect to nuclear empirical
    parameters must survive the full composition.  Forward-mode (jacfwd) is
    the supported TOV sensitivity path.  We require sign agreement and <25%
    relative error; measured on ldas-pcdev13 (jax 0.10.1): AD -5.17e-6 vs FD
    -6.32e-6 Msun/MeV (18%).  On sporcsubmit (jax 0.10.1, 2026-10-12): AD
    -2.0319e-5 vs FD -2.0366e-5 (-0.23%) before steps stopped at the table
    nodes, -2.0346e-5 vs -2.0366e-5 (-0.10%) after (test_tov_jump_ts.py).
    """
    from nmbackend.eos import MetaModelEOS
    from nmbackend.tov import build_family
    from nmbackend.tov.family import default_hc_grid

    mm = MetaModelEOS()

    def fam_of(L):
        t = mm.eos_table({"L_sym": L})
        return build_family(t, default_hc_grid(t, n_models=140),
                            rtol=1e-9, atol=1e-11, n_steps=800)

    def mej(L):
        return ejecta_from_family(fam_of(L), 1.35, 1.30).m_dyn.value

    g = float(jax.jacfwd(mej)(60.0))
    fd = (float(mej(62.0)) - float(mej(58.0))) / 4.0
    assert np.isfinite(g)
    assert g * fd > 0.0
    assert abs(g - fd) <= 0.25 * abs(fd)
