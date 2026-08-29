"""Round-trip + dispatch tests for the compressed PCA sequence representation.

The compressed ``pca_hc/1`` file must reconstruct the per-EOS Lambda(m), R(m), and
M_max of the tabular sequence to good accuracy with a handful of components, and
``open_sequence`` must dispatch to it transparently.  We build the (expensive) tabular
sequence ONCE, then compress it without re-running TOV via ``write_pca_from_nss``.
"""
import json
import numpy as np
import pytest

pytest.importorskip("h5py")

from nmbackend.eos import MetaModelEOS
from nmbackend.rift import (write_nss, write_pca_from_nss, open_sequence,
                            NSSequence, PCASequence, compression_ratio)

_MREF = 1.4


@pytest.fixture(scope="module")
def seqs(tmp_path_factory):
    mm = MetaModelEOS()
    params = [{"L_sym": L, "cs2_tail": c}
              for L in (40.0, 55.0, 70.0) for c in (0.30, 0.45, 0.60, 0.75)]
    tables = [mm.eos_table(p) for p in params]
    d = tmp_path_factory.mktemp("compressed")
    nss = str(d / "seq.nss.h5")
    pca = str(d / "seq.pca.h5")
    write_nss(nss, tables, params=params, n_points=60)
    write_pca_from_nss(nss, pca, n_components=6)
    return nss, pca


def test_open_sequence_dispatches_to_pca(seqs):
    _, pca = seqs
    seq = open_sequence(pca)
    assert isinstance(seq, PCASequence)


def test_pca_roundtrip_matches_tabular(seqs):
    nss, pca = seqs
    ref, comp = NSSequence(nss), PCASequence(pca)
    assert comp.n_eos == ref.n_eos
    lam_ref = np.array([ref.lambda_of_m(_MREF, e) for e in range(ref.n_eos)])
    lam_cmp = np.array([comp.lambda_of_m(_MREF, e) for e in range(comp.n_eos)])
    ok = np.isfinite(lam_ref) & np.isfinite(lam_cmp)
    rel = np.abs(lam_cmp[ok] - lam_ref[ok]) / lam_ref[ok]
    assert np.median(rel) < 0.02 and np.max(rel) < 0.06
    r_ref = np.array([ref.radius_of_m(_MREF, e) for e in range(ref.n_eos)])
    r_cmp = np.array([comp.radius_of_m(_MREF, e) for e in range(comp.n_eos)])
    ok = np.isfinite(r_ref) & np.isfinite(r_cmp)
    assert np.max(np.abs(r_cmp[ok] - r_ref[ok])) < 0.05      # < 50 m
    mm_ref = np.array([ref.m_max(e) for e in range(ref.n_eos)])
    mm_cmp = np.array([comp.m_max(e) for e in range(comp.n_eos)])
    assert np.max(np.abs(mm_cmp - mm_ref)) < 0.02


def test_pca_ordering_statistic(seqs):
    _, pca = seqs
    comp = PCASequence(pca)
    vals = comp.ordering_values(_MREF, "Lambda")
    assert vals.shape == (comp.n_eos,) and np.all(np.isfinite(vals))


def test_compression_ratio_grows_with_neos():
    assert compression_ratio(10000, 80, 6) > compression_ratio(100, 80, 6) > 1.0


def test_pca_primary_branch_does_not_mix_disconnected_stable_branches(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "twins.pca.h5"
    mass = np.array([1.0, 2.0, 1.8, 1.6, 1.9, 2.1])
    radius = np.array([13.0, 11.0, 10.8, 10.0, 9.5, 9.0])
    lam = np.array([600.0, 100.0, 80.0, 60.0, 35.0, 20.0])
    with h5py.File(path, "w") as f:
        f.attrs["representation"] = "pca_hc/1"
        f.attrs["schema_version"] = "nmbackend.pca/1"
        f.attrs["channels"] = json.dumps(["M", "R", "logLambda"])
        f.create_dataset("mean", data=np.stack([mass, radius, np.log(lam)]))
        f.create_dataset("components", data=np.zeros((3, 1, mass.size)))
        f.create_dataset("coeffs", data=np.zeros((1, 3, 1)))

    seq = PCASequence(path)
    assert seq.n_stable_branches(0) == 2
    assert seq.stable_branch_counts.tolist() == [2]
    assert seq.m_max(0) == pytest.approx(2.0)
    assert seq.radius_of_m(1.8, 0) == pytest.approx(11.4)
    expected_lambda = np.exp(np.interp(1.8, [1.0, 2.0], np.log([600.0, 100.0])))
    assert seq.lambda_of_m(1.8, 0) == pytest.approx(expected_lambda)
