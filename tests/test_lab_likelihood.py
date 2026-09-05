"""MB-N2 density-tagged lab likelihood: analytic Gaussian forms,
sigma_model marginalization-vs-fixed consistency, scenario-ladder round-trip."""
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


# ----- sigma_model marginalization vs fixed -----

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
    marg = float(_single(sigma_model=prior)())
    fixed = [float(_single(sigma_model=s)()) for s in prior.grid]
    assert min(fixed) <= marg <= max(fixed)


def test_half_normal_prior_normalized_and_marginalizes():
    prior = SigmaModelPrior.half_normal(2.0)
    assert np.exp(prior.log_weights).sum() == pytest.approx(1.0, rel=1e-12)
    val = float(_single(sigma_model=prior)())
    assert np.isfinite(val)


def test_marginalized_gradient_finite():
    # downstream Fisher/AD consumers differentiate lnL w.r.t. the NEPs
    like = _single(sigma_model=SigmaModelPrior.half_normal(2.0))
    g = jax.grad(lambda L: like.ln_likelihood_marginalized({"L_sym": L}))(60.0)
    eps = 1e-4
    fd = (float(like.ln_likelihood_marginalized({"L_sym": 60.0 + eps}))
          - float(like.ln_likelihood_marginalized({"L_sym": 60.0 - eps}))) / (2 * eps)
    assert float(g) == pytest.approx(fd, rel=1e-6)


# ----- beta-equilibrium caveat (MB-N5) -----

def test_warns_above_matching_density():
    import warnings
    c = DensityTaggedConstraint(n=2.5 * NSAT, mu=30.0, sigma=10.0, name="above")
    with pytest.warns(UserWarning, match="beta-equilibrated"):
        DensityTaggedPressureLikelihood([c])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DensityTaggedPressureLikelihood([c], warn_above_match=False)


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


def test_ladder_copy_carries_provenance_header():
    from importlib import resources
    text = (resources.files("nmbackend") / "data"
            / "frib_scenario_ladder.yaml").read_text()
    assert "COPY" in text and "origin" in text
    assert "EVERY NUMBER IN THIS FILE IS AN ASSUMPTION" in text


def test_gaussian_from_scenario_roundtrip():
    ladder = load_scenario_ladder()
    assert gaussian_from_scenario(scenario_by_name(ladder, "GW-only")) is None
    g = gaussian_from_scenario(scenario_by_name(ladder, "+FRIB-2028"))
    # peak at the fiducial means; widths are the ladder numbers
    peak = float(g({"E_sym": DEFAULT_PARAMS["E_sym"],
                    "L_sym": DEFAULT_PARAMS["L_sym"]}))
    expect_peak = (norm.logpdf(0.0, 0.0, 2.0) + norm.logpdf(0.0, 0.0, 22.0))
    assert peak == pytest.approx(expect_peak, rel=1e-12)


def test_density_tagged_from_scenario_roundtrip():
    ladder = load_scenario_ladder()
    assert density_tagged_from_scenario(scenario_by_name(ladder, "+FRIB-2028")) is None
    like = density_tagged_from_scenario(scenario_by_name(ladder, "+FRIB400+Ksym"))
    (c,) = like.constraints
    assert c.n == pytest.approx(2.0 * NSAT)
    assert c.mu == pytest.approx(float(psym(2.0 * NSAT)))
    assert c.sigma == pytest.approx(0.30 * abs(c.mu))
    # forecast injection: fiducial model sits at the peak of the likelihood
    assert float(like()) == pytest.approx(
        norm.logpdf(0.0, 0.0, c.sigma), rel=1e-12)
