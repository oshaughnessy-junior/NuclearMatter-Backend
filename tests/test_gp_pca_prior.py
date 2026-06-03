"""Adaptive Gaussian prior over PCA sequence space: fit, sample, refine, converge.

The decisive test: the iterative-refinement loop must CONCENTRATE the prior on a
synthetic evidence target (lnZ peaked at Lambda(1.4) = target) -- the round-over-
round posterior-predictive spread of Lambda(1.4) shrinks and its center moves to
the target.  This is the RIFT-GP adaptive-prior primitive, evidence_fn abstracted.
"""
import numpy as np
import pytest

pytest.importorskip("h5py")

from nmbackend.eos import MetaModelEOS
from nmbackend.rift import write_nss, write_pca_from_nss, open_sequence, PCASequence
from nmbackend.priors import CoefficientPrior, write_pca_draws, iterative_refinement
from nmbackend.rift.sequence import _stable_rising


@pytest.fixture(scope="module")
def pca_file(tmp_path_factory):
    mm = MetaModelEOS()
    params = [{"L_sym": L, "cs2_tail": c}
              for L in (35.0, 50.0, 65.0, 80.0) for c in (0.30, 0.45, 0.60)]
    tables = [mm.eos_table(p) for p in params]
    d = tmp_path_factory.mktemp("gp_pca")
    nss = str(d / "train.nss.h5")
    pca = str(d / "train.pca.h5")
    write_nss(nss, tables, params=params, n_points=60)
    write_pca_from_nss(nss, pca, n_components=6)
    return pca


def _lambda14(prior, coeff):
    M, R, Lam = prior.reconstruct(coeff)
    stable = np.concatenate([[True], np.diff(M) > 0]).astype(float)
    Mb, Rb, Lb = _stable_rising(M, R, Lam, stable)
    if Mb.size < 2 or not (Mb[0] <= 1.4 <= Mb[-1]):
        return np.nan
    return float(np.exp(np.interp(1.4, Mb, np.log(np.maximum(Lb, 1e-30)))))


def test_fit_and_sample_covers_training(pca_file):
    prior = CoefficientPrior.from_pca_file(pca_file)
    rng = np.random.default_rng(1)
    coeffs, keep = prior.sample_sequences(64, rng)
    assert keep > 0.5                       # most raw draws are valid sequences
    lams = np.array([_lambda14(prior, c) for c in coeffs])
    lams = lams[np.isfinite(lams)]
    train = PCASequence(pca_file)
    lam_train = np.array([train.lambda_of_m(1.4, e) for e in range(train.n_eos)])
    # samples span (at least) the central range of the training family
    assert np.nanmin(lams) < np.nanpercentile(lam_train, 30)
    assert np.nanmax(lams) > np.nanpercentile(lam_train, 70)


def test_adaptive_refinement_concentrates_on_target(pca_file):
    prior0 = CoefficientPrior.from_pca_file(pca_file)
    rng = np.random.default_rng(7)
    target, width = 400.0, 30.0

    def evidence_fn(prior, coeffs):
        lams = np.array([_lambda14(prior, c) for c in coeffs])
        return -0.5 * ((lams - target) / width) ** 2     # synthetic lnZ

    prior, hist = iterative_refinement(prior0, evidence_fn, rng,
                                       n_rounds=4, n_per_round=96,
                                       beta=1.0, shrink=0.2)
    # round-0 vs final-round draw statistics of Lambda(1.4)
    def stats(h):
        lams = np.array([_lambda14(prior0, c) for c in h["coeffs"]])
        lams = lams[np.isfinite(lams)]
        return np.mean(lams), np.std(lams)
    m0, s0 = stats(hist[0])
    mN, sN = stats(hist[-1])
    assert abs(mN - target) < abs(m0 - target)           # center moves to target
    assert abs(mN - target) < 60.0                       # and lands near it
    assert sN < 0.7 * s0                                 # spread shrinks
    assert prior.generation == 4
    assert all(np.isfinite(h["ess"]) and h["ess"] > 1 for h in hist)


def test_draws_roundtrip_as_pca_file(pca_file, tmp_path):
    prior = CoefficientPrior.from_pca_file(pca_file)
    rng = np.random.default_rng(3)
    coeffs, _ = prior.sample_sequences(10, rng)
    out = str(tmp_path / "draws.pca.h5")
    write_pca_draws(out, prior, coeffs)
    seq = open_sequence(out)
    assert isinstance(seq, PCASequence) and seq.n_eos == 10
    vals = seq.ordering_values(1.4, "Lambda")
    assert np.isfinite(vals).sum() >= 8
