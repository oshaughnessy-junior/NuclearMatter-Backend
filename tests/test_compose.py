"""CompOSE / tabular EOS ingestion: round-trip exactness + cross-code on a table."""
import numpy as np
import tempfile

from nmbackend.eos import RMFTEOS
from nmbackend.eos.compose import (write_compose, load_compose, read_compose,
                                   eos_table_from_arrays, load_simple)
from nmbackend.tov.family import (build_family, default_hc_grid, m_max,
                                  radius_of_mass, lambda_of_mass)
from nmbackend.tov.solver import solve_structure
from nmbackend.reference.scipy_tov import solve_reference


def _rmft_arrays():
    t = RMFTEOS().eos_table()
    n = np.asarray(t.n)
    m = n > 3e-4
    return n[m], np.asarray(t.eps_mevfm3)[m], np.asarray(t.p_mevfm3)[m]


def test_compose_roundtrip_arrays_exact():
    n, eps, p = _rmft_arrays()
    d = tempfile.mkdtemp()
    write_compose(d, n, eps, p)
    n2, e2, p2, _ = read_compose(d)
    assert np.allclose(n2, n)
    assert np.max(np.abs(e2 - eps) / eps) < 1e-8
    assert np.max(np.abs(p2 - p) / np.maximum(p, 1e-6)) < 1e-8


def test_compose_roundtrip_tov_matches():
    # an ingested CompOSE table must give the SAME M-R-Lambda as the original EOS
    rm = RMFTEOS()
    t0 = rm.eos_table()
    n, eps, p = _rmft_arrays()
    d = tempfile.mkdtemp()
    write_compose(d, n, eps, p)
    t1 = load_compose(d)
    f0 = build_family(t0, default_hc_grid(t0, n_models=45))
    f1 = build_family(t1, default_hc_grid(t1, n_models=45))
    assert abs(float(m_max(f0)) - float(m_max(f1))) < 1e-3
    assert abs(float(radius_of_mass(f0, 1.4)) - float(radius_of_mass(f1, 1.4))) < 1e-2
    assert abs(float(lambda_of_mass(f0, 1.4)) - float(lambda_of_mass(f1, 1.4))) < 1.0


def test_ingested_table_cross_code():
    # cross-code: diffrax vs the independent SciPy (H,beta) solver on the ingested
    # tabular EOS agree to the internal floor (~m / <0.1%)
    n, eps, p = _rmft_arrays()
    t = eos_table_from_arrays(n, eps, p)
    hc = 0.25 * float(t.h[-1])
    a = solve_structure(t, hc, backend="diffrax", rtol=1e-9, atol=1e-11)
    b = solve_reference(t, hc, rtol=1e-10, atol=1e-12)
    assert abs(float(a.R_km) - b.R_km) * 1000.0 < 30.0       # metres
    assert abs(float(a.Lambda) - b.Lambda) / b.Lambda < 0.01


def test_load_simple_columns():
    n, eps, p = _rmft_arrays()
    import os
    f = os.path.join(tempfile.mkdtemp(), "eos.dat")
    np.savetxt(f, np.column_stack([n, eps, p]), header="n_B eps p")
    t = load_simple(f)
    assert np.all(np.diff(np.asarray(t.h)) > 0)
    assert float(t.n[0]) <= n[0] + 1e-12
