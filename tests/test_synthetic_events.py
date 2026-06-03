"""Genie-style synthetic multi-event evidence: recipe sanity + adaptive recovery.

The decisive test: a multi-event synthetic set built from a known 'true' EOS draw
must let the adaptive loop recover the truth -- the refined prior's draws
concentrate around the truth's LambdaTilde surface.
"""
import numpy as np
import pytest

pytest.importorskip("h5py")

from nmbackend.eos import MetaModelEOS
from nmbackend.rift import write_nss, write_pca_from_nss
from nmbackend.priors import CoefficientPrior, iterative_refinement
from nmbackend.priors.gp_pca import sequence_lambda_of_m
from nmbackend.inference import lambda_tilde, draw_bns_events, SyntheticTidalEvents


@pytest.fixture(scope="module")
def prior_and_truth(tmp_path_factory):
    mm = MetaModelEOS()
    params = [{"L_sym": L, "cs2_tail": c}
              for L in (35.0, 50.0, 65.0, 80.0) for c in (0.30, 0.45, 0.60)]
    tables = [mm.eos_table(p) for p in params]
    d = tmp_path_factory.mktemp("synth")
    nss = str(d / "t.nss.h5"); pca = str(d / "t.pca.h5")
    write_nss(nss, tables, params=params, n_points=60)
    write_pca_from_nss(nss, pca, n_components=6)
    prior = CoefficientPrior.from_pca_file(pca)
    # truth = one in-family draw (coefficient vector of training EOS #5)
    from nmbackend.rift import PCASequence
    truth_coeff = PCASequence(pca).coeffs[5]
    return prior, truth_coeff


def test_lambda_tilde_limits():
    # equal masses, equal lambdas -> LambdaTilde == lambda
    assert abs(lambda_tilde(1.4, 1.4, 300.0, 300.0) - 300.0) < 1e-9
    # heavier primary dominates appropriately (positive, monotone in lam1)
    assert lambda_tilde(1.6, 1.2, 200.0, 800.0) > 0
    assert (lambda_tilde(1.4, 1.4, 400.0, 300.0)
            > lambda_tilde(1.4, 1.4, 300.0, 300.0))


def test_event_drawing(prior_and_truth):
    prior, truth_coeff = prior_and_truth
    truth = lambda m: sequence_lambda_of_m(prior, truth_coeff, m)
    rng = np.random.default_rng(2)
    evs = draw_bns_events(rng, 8, truth, sigma_lamtilde=60.0)
    assert len(evs) >= 6
    for ev in evs:
        assert ev["m1"] >= ev["m2"] and ev["sigma"] > 0
        assert np.isfinite(ev["x_obs"]) and np.isfinite(ev["x_true"])


def test_adaptive_loop_recovers_truth(prior_and_truth):
    prior0, truth_coeff = prior_and_truth
    truth = lambda m: sequence_lambda_of_m(prior0, truth_coeff, m)
    rng = np.random.default_rng(11)
    events = draw_bns_events(rng, 6, truth, sigma_lamtilde=50.0)
    ev_fn = SyntheticTidalEvents(events)
    prior, hist = iterative_refinement(prior0, ev_fn, rng, n_rounds=4,
                                       n_per_round=96, shrink=0.2)
    # posterior-predictive Lambda(1.4) of the refined prior vs truth
    lam_true = truth(1.4)
    coeffs, _ = prior.sample_sequences(60, rng)
    lams = np.array([sequence_lambda_of_m(prior, c, 1.4) for c in coeffs])
    lams = lams[np.isfinite(lams)]
    lam0s = np.array([sequence_lambda_of_m(prior0, c, 1.4)
                      for c in prior0.sample_sequences(60, rng)[0]])
    lam0s = lam0s[np.isfinite(lam0s)]
    assert abs(np.mean(lams) - lam_true) < abs(np.mean(lam0s) - lam_true) + 10.0
    assert np.std(lams) < 0.6 * np.std(lam0s)        # multi-event contraction
    assert abs(np.mean(lams) - lam_true) < 2.5 * np.std(lams) + 20.0
