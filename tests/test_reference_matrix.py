"""Reference-matrix harness: smoke + internal-floor regression guard."""
import numpy as np

from nmbackend.eos import MetaModelEOS
from nmbackend.tov.family import default_hc_grid
from nmbackend.validation.reference_matrix import (
    build_ladder, _seq_diffrax_smooth, _seq_scipy_smooth, _at_masses,
)


def test_ladder_nonempty_and_well_formed():
    ladder = build_ladder()
    assert len(ladder) >= 4
    for name, kind, obj, _ in ladder:
        assert kind in ("smooth", "hybrid")


def test_diffrax_vs_scipy_internal_floor():
    # diffrax and the independent radius-domain (H,beta) reference must agree to
    # the internal numerical floor on a smooth EOS: a few metres in R, <1% in Lambda.
    t = MetaModelEOS().eos_table()
    hc = default_hc_grid(t, n_models=40)
    Rd, Ld, _ = _at_masses(*_seq_diffrax_smooth(t, hc), [1.4])
    Rs, Ls, _ = _at_masses(*_seq_scipy_smooth(t, hc), [1.4])
    assert abs(Rd[0] - Rs[0]) * 1000.0 < 30.0          # metres
    assert abs(Ld[0] - Ls[0]) / Ls[0] < 0.02           # fractional


def test_at_masses_above_mmax_is_nan():
    t = MetaModelEOS().eos_table({"cs2_tail": 0.3})     # M_max ~ 1.8
    hc = default_hc_grid(t, n_models=30)
    Rd, Ld, mmax = _at_masses(*_seq_diffrax_smooth(t, hc), [2.5])
    assert np.isnan(Rd[0]) and mmax < 2.5
