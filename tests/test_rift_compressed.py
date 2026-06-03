"""Round-trip + dispatch tests for the compressed PCA sequence representation.

The compressed ``pca_hc/1`` file must reconstruct the per-EOS Lambda(m), R(m), and
M_max of the tabular sequence to good accuracy with a handful of components, and
``open_sequence`` must dispatch to it transparently.  We build the (expensive) tabular
sequence ONCE, then compress it without re-running TOV via ``write_pca_from_nss``.
"""
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
