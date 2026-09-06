"""MB-N2 density-tagged lab likelihood: analytic Gaussian forms,
sigma_model marginalization-vs-fixed consistency, scenario-ladder round-trip,
and regressions for the 2026-09-05 adversarial-review findings."""
import warnings

import numpy as np
import pytest
from scipy.stats import norm, multivariate_normal

import jax

from nmbackend.eos.metamodel import DEFAULT_PARAMS
from nmbackend.lab_likelihood import (
    psym,
    DensityTaggedConstraint,
    DensityTaggedPressureLikelihood,
    SigmaModelPrior,
    CorrelatedSymGaussian,
    load_scenario_ladder,
    scenario_by_name,
    gaussian_from_scenario,
    density_tagged_from_scenario,
)

NSAT = DEFAULT_PARAMS["n_sat"]


# ----- P_sym(n) against the closed-form expansion derivative -----

def test_psym_saturation_analytic():
    # P_sym(n_sat) = n_sat * L_sym / 3 exactly (only the L term survives at x=0)
    for L in (30.0, 60.0, 90.0):
        got = float(psym(NSAT, {"L_sym": L}))
        assert got == pytest.approx(NSAT * L / 3.0, rel=1e-12)


def test_psym_matches_closed_form_off_saturation():
    # S = E + L x + K x^2/2 + Q x^3/6  =>  P_sym = n^2 (L + K x + Q x^2/2)/(3 n_sat)
    p = dict(E_sym=31.0, L_sym=70.0, K_sym=-120.0, Q_sym=200.0)
    n = np.array([0.5, 1.0, 1.5, 2.0]) * NSAT
    x = (n - NSAT) / (3.0 * NSAT)
    expect = n**2 * (p["L_sym"] + p["K_sym"] * x + 0.5 * p["Q_sym"] * x**2) / (3.0 * NSAT)
    got = np.asarray(psym(n, p))
    assert got == pytest.approx(expect, rel=1e-12)


def test_psym_shape_in_shape_out():
    # scalar -> 0-d; 1-D -> 1-D; 2-D -> 2-D (used to raise deep inside jax.grad)
    assert np.ndim(psym(NSAT)) == 0
    n2 = np.array([[0.5, 1.0], [1.5, 2.0]]) * NSAT
    out = np.asarray(psym(n2))
    assert out.shape == n2.shape
    assert out == pytest.approx(np.asarray(psym(n2.ravel())).reshape(2, 2))


def test_unknown_nep_key_raises():
    # a typo'd key used to be silently ignored (constant lnL, zero gradient)
    with pytest.raises(ValueError, match="unknown EOS parameter"):
        psym(NSAT, {"Lsym": 60.0})
    like = _single()
    with pytest.raises(ValueError, match="unknown EOS parameter"):
        like.ln_likelihood({"Lsym": 60.0})


# ----- Gaussian forms, analytic -----

def _single(mu=10.0, sigma=3.0, n=1.5 * NSAT, **kw):
    c = DensityTaggedConstraint(n=n, mu=mu, sigma=sigma, name="test")
    return DensityTaggedPressureLikelihood([c], **kw)


def test_density_tagged_gaussian_analytic():
    like = _single(sigma_model=2.0)
    ps = float(like.psym_tagged()[0])
    expect = norm.logpdf(ps, loc=10.0, scale=np.hypot(3.0, 2.0))
    assert float(like()) == pytest.approx(expect, rel=1e-12)


def test_density_tagged_sum_over_constraints():
    cs = [DensityTaggedConstraint(n=NSAT, mu=8.0, sigma=2.0),
          DensityTaggedConstraint(n=1.5 * NSAT, mu=20.0, sigma=5.0)]
    like = DensityTaggedPressureLikelihood(cs, sigma_model=1.0)
    ps = np.asarray(like.psym_tagged())
    expect = (norm.logpdf(ps[0], 8.0, np.hypot(2.0, 1.0))
              + norm.logpdf(ps[1], 20.0, np.hypot(5.0, 1.0)))
    assert float(like()) == pytest.approx(expect, rel=1e-12)


def test_correlated_gaussian_matches_scipy():
    g = CorrelatedSymGaussian.from_sigmas(32.0, 60.0, 2.0, 20.0, rho=0.7)
    params = {"E_sym": 33.5, "L_sym": 85.0}
    cov = [[4.0, 0.7 * 2.0 * 20.0], [0.7 * 2.0 * 20.0, 400.0]]
    expect = multivariate_normal.logpdf([33.5, 85.0], mean=[32.0, 60.0], cov=cov)
    assert float(g(params)) == pytest.approx(expect, rel=1e-12)


def test_correlated_gaussian_rho_zero_is_diagonal():
    g = CorrelatedSymGaussian.from_sigmas(32.0, 60.0, 2.0, 20.0, rho=0.0)
    params = {"E_sym": 30.0, "L_sym": 100.0}
    expect = norm.logpdf(30.0, 32.0, 2.0) + norm.logpdf(100.0, 60.0, 20.0)
    assert float(g(params)) == pytest.approx(expect, rel=1e-12)


def test_from_sigmas_rejects_negative_sigma():
    # a negative sigma survives squaring and used to silently flip rho's sign
    with pytest.raises(ValueError, match="sigma_E and sigma_L"):
        CorrelatedSymGaussian.from_sigmas(32.0, 60.0, -2.0, 20.0, rho=0.7)


def test_correlated_gaussian_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        CorrelatedSymGaussian([32.0, 60.0], [[4.0, np.nan], [np.nan, 400.0]])


# ----- sigma_model validation and marginalization vs fixed -----

def test_sigma_model_scalar_validation():
    with pytest.raises(ValueError, match="sigma_model"):
        _single(sigma_model=-1.0)
    # numpy/jax scalars used to bypass the isinstance guard entirely
    with pytest.raises(ValueError, match="sigma_model"):
        _single(sigma_model=np.float32(-1.0))


def test_direct_prior_construction_is_normalized():
    # direct SigmaModelPrior(grid, log_weights) used to skip normalization,
    # silently offsetting every lnL (and any Bayes factor) by a constant
    direct = SigmaModelPrior(grid=(0.3,), log_weights=(5.0,))
    assert np.exp(direct.log_weights).sum() == pytest.approx(1.0, rel=1e-12)
    assert float(_single(sigma_model=direct)()) == pytest.approx(
        float(_single(sigma_model=0.3)()), rel=1e-12)


def test_prior_rejects_degenerate_n():
    # n=1 used to silently collapse uniform/half_normal to a delta at lo
    with pytest.raises(ValueError, match="n >= 2"):
        SigmaModelPrior.uniform(0.0, 2.0, n=1)
    with pytest.raises(ValueError, match="n >= 2"):
        SigmaModelPrior.half_normal(2.0, n=1)
    with pytest.raises(ValueError, match="n >= 2"):
        SigmaModelPrior.from_logpdf(lambda s: 0.0, 0.0, 2.0, n=1)
    with pytest.raises(ValueError, match="lo < hi"):
        SigmaModelPrior.from_logpdf(lambda s: 0.0, 1.0, 0.0)


def test_marginalized_delta_prior_equals_fixed():
    fixed = _single(sigma_model=2.0)
    marg = _single(sigma_model=SigmaModelPrior.fixed(2.0))
    assert float(marg()) == pytest.approx(float(fixed()), rel=1e-12)


def test_marginalized_narrow_prior_consistent_with_fixed():
    s0 = 2.0
    fixed = _single(sigma_model=s0)
    marg = _single(sigma_model=SigmaModelPrior.uniform(s0 - 1e-4, s0 + 1e-4, n=65))
    assert float(marg()) == pytest.approx(float(fixed()), rel=1e-8)


def test_marginalized_bracketed_by_fixed_extremes():
    # the marginalized likelihood is a prior-weighted average of the fixed-sigma
    # likelihood, so it must lie between the min and max of lnL(sigma) over the
    # prior support (note lnL(sigma) is NOT monotone: it peaks where the total
    # variance matches the residual, so endpoint values alone do not bracket it)
    prior = SigmaModelPrior.uniform(0.5, 6.0)
    like = _single(sigma_model=prior)
    marg = float(like())
    fixed = [float(like.ln_likelihood(sigma_model=s)) for s in prior.grid]
    assert min(fixed) <= marg <= max(fixed)


def test_half_normal_prior_normalized_and_marginalizes():
    prior = SigmaModelPrior.half_normal(2.0)
    assert np.exp(prior.log_weights).sum() == pytest.approx(1.0, rel=1e-12)
    val = float(_single(sigma_model=prior)())
    assert np.isfinite(val)


def test_prior_fixed_mismatch_error_messages():
    fixed = _single(sigma_model=2.0)
    marg = _single(sigma_model=SigmaModelPrior.half_normal(2.0))
    with pytest.raises(TypeError, match="marginalized"):
        marg.ln_likelihood()
    with pytest.raises(TypeError, match="SigmaModelPrior"):
        fixed.ln_likelihood_marginalized()
    with pytest.raises(TypeError, match="marginalized"):
        fixed.ln_likelihood(sigma_model=SigmaModelPrior.fixed(1.0))
    with pytest.raises(TypeError, match="SigmaModelPrior"):
        marg.ln_likelihood_marginalized(prior=0.3)


def test_marginalized_gradient_finite():
    # downstream Fisher/AD consumers differentiate lnL w.r.t. the NEPs
    like = _single(sigma_model=SigmaModelPrior.half_normal(2.0))
    g = jax.grad(lambda L: like.ln_likelihood_marginalized({"L_sym": L}))(60.0)
    eps = 1e-4
    fd = (float(like.ln_likelihood_marginalized({"L_sym": 60.0 + eps}))
          - float(like.ln_likelihood_marginalized({"L_sym": 60.0 - eps}))) / (2 * eps)
    assert float(g) == pytest.approx(fd, rel=1e-6)


def test_constraint_validation():
    with pytest.raises(ValueError, match="finite"):
        DensityTaggedConstraint(n=NSAT, mu=np.nan, sigma=1.0)
    with pytest.raises(ValueError, match="underflow|sigma"):
        DensityTaggedConstraint(n=NSAT, mu=1.0, sigma=1e-200)


# ----- beta-equilibrium caveat (MB-N5) -----

def test_warns_at_and_above_matching_density():
    c_above = DensityTaggedConstraint(n=2.5 * NSAT, mu=30.0, sigma=10.0, name="above")
    with pytest.warns(UserWarning, match="beta-equilibrated"):
        DensityTaggedPressureLikelihood([c_above])
    # the boundary itself is IN the beta-eq sector (bridge keeps n < n_L), so
    # a tag at exactly 2 n_sat must warn too
    c_at = DensityTaggedConstraint(n=2.0 * NSAT, mu=30.0, sigma=10.0, name="at")
    with pytest.warns(UserWarning, match="beta-equilibrated"):
        DensityTaggedPressureLikelihood([c_at])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DensityTaggedPressureLikelihood([c_above], warn_above_match=False)


def test_n_beta_eq_override_moves_threshold():
    # bridges drawn with non-default n_L_fac have a different boundary than the
    # metamodel's n_match_fac; the explicit override must control the warning
    c = DensityTaggedConstraint(n=1.6 * NSAT, mu=10.0, sigma=3.0, name="mid")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DensityTaggedPressureLikelihood([c])          # below default 2 n_sat
    with pytest.warns(UserWarning, match="beta-equilibrated"):
        DensityTaggedPressureLikelihood([c], n_beta_eq=1.5 * NSAT)


# ----- fiducial (eos_params) plumbing -----

def test_eos_params_is_evaluation_base():
    # the construction-time fiducial must be the evaluation default, and
    # per-call params merge over it
    c = DensityTaggedConstraint(n=NSAT, mu=1.0, sigma=1.0)
    like = DensityTaggedPressureLikelihood([c], eos_params={"L_sym": 40.0})
    assert float(like.psym_tagged()[0]) == pytest.approx(NSAT * 40.0 / 3.0, rel=1e-12)
    assert float(like.psym_tagged({"L_sym": 80.0})[0]) == pytest.approx(
        NSAT * 80.0 / 3.0, rel=1e-12)


# ----- scenario-ladder round-trip -----

def test_ladder_loads_and_matches_source_numbers():
    ladder = load_scenario_ladder()
    assert ladder["version"] == 1
    names = [s["name"] for s in ladder["scenarios"]]
    assert names == ["GW-only", "+FRIB-2028", "+FRIB400", "+FRIB400+Ksym"]
    near = scenario_by_name(ladder, "+FRIB-2028")
    assert near["lab_sigma"] == {"L_sym": 22.0, "E_sym": 2.0}
    four = scenario_by_name(ladder, "Four")           # tag lookup
    assert four["lab_sigma"] == {"L_sym": 15.0, "E_sym": 1.5}
    fourk = scenario_by_name(ladder, "+FRIB400+Ksym")
    assert fourk["psym2n0_frac"] == pytest.approx(0.30)
    # every scenario must be labeled an assumption (anti-oversell guard)
    assert all(s["assumption"] is True for s in ladder["scenarios"])
    with pytest.raises(KeyError, match="tag"):
        scenario_by_name(ladder, "nope")


def test_ladder_copy_carries_provenance_header():
    from importlib import resources
    text = (resources.files("nmbackend") / "data"
            / "frib_scenario_ladder.yaml").read_text(encoding="utf-8")
    assert "COPY" in text and "origin" in text
    assert "EVERY NUMBER IN THIS FILE IS AN ASSUMPTION" in text
    assert "P_sym" in text        # the psym2n0_frac interpretation note


def test_gaussian_from_scenario_roundtrip():
    ladder = load_scenario_ladder()
    assert gaussian_from_scenario(scenario_by_name(ladder, "GW-only")) is None
    g = gaussian_from_scenario(scenario_by_name(ladder, "+FRIB-2028"))
    # peak at the fiducial means; widths are the ladder numbers
    peak = float(g({"E_sym": DEFAULT_PARAMS["E_sym"],
                    "L_sym": DEFAULT_PARAMS["L_sym"]}))
    expect_peak = (norm.logpdf(0.0, 0.0, 2.0) + norm.logpdf(0.0, 0.0, 22.0))
    assert peak == pytest.approx(expect_peak, rel=1e-12)
    # a custom fiducial must move the means (claim-C3 centering consistency)
    g40 = gaussian_from_scenario(scenario_by_name(ladder, "+FRIB-2028"),
                                 eos_params={"L_sym": 40.0})
    assert float(g40({"E_sym": DEFAULT_PARAMS["E_sym"], "L_sym": 40.0})) == \
        pytest.approx(expect_peak, rel=1e-12)


def test_gaussian_from_scenario_warns_on_dropped_keys():
    # +FRIB400+Ksym carries K_sym, which the 2x2 form cannot encode; a silent
    # drop made rungs 3 and 4 bit-identical with no signal
    ladder = load_scenario_ladder()
    with pytest.warns(UserWarning, match="DROPPED"):
        gaussian_from_scenario(scenario_by_name(ladder, "+FRIB400+Ksym"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaussian_from_scenario(scenario_by_name(ladder, "+FRIB400"))


def test_density_tagged_from_scenario_roundtrip():
    ladder = load_scenario_ladder()
    assert density_tagged_from_scenario(scenario_by_name(ladder, "+FRIB-2028")) is None
    # the 2 n_sat tag sits AT the beta-eq boundary, so the MB-N5 warning fires
    with pytest.warns(UserWarning, match="beta-equilibrated"):
        like = density_tagged_from_scenario(scenario_by_name(ladder, "+FRIB400+Ksym"))
    (c,) = like.constraints
    assert c.n == pytest.approx(2.0 * NSAT)
    assert c.mu == pytest.approx(float(psym(2.0 * NSAT)))
    assert c.sigma == pytest.approx(0.30 * c.mu)
    # forecast injection: fiducial model sits at the peak of the likelihood
    assert float(like()) == pytest.approx(
        norm.logpdf(0.0, 0.0, c.sigma), rel=1e-12)


def test_density_tagged_from_scenario_custom_fiducial_peaks_there():
    # with eos_params stored as the evaluation base, the peak property must
    # hold off-default too (it used to silently break)
    ladder = load_scenario_ladder()
    sc = scenario_by_name(ladder, "+FRIB400+Ksym")
    like = density_tagged_from_scenario(sc, eos_params={"L_sym": 80.0},
                                        warn_above_match=False)
    (c,) = like.constraints
    assert float(like()) == pytest.approx(
        norm.logpdf(0.0, 0.0, c.sigma), rel=1e-12)


def test_density_tagged_from_scenario_rejects_nonpositive_psym():
    # sigma = frac * mu degenerates (near-delta, then crash) as P_sym(2n0)
    # crosses zero; must fail loudly with an explanation, not mid-sweep
    ladder = load_scenario_ladder()
    sc = scenario_by_name(ladder, "+FRIB400+Ksym")
    with pytest.raises(ValueError, match="not positive"):
        density_tagged_from_scenario(sc, eos_params={"L_sym": 20.0},
                                     warn_above_match=False)
