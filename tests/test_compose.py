"""CompOSE / tabular EOS ingestion: round-trip exactness + cross-code on a table."""
import numpy as np
import tempfile
import warnings

from nmbackend.eos import RMFTEOS
from nmbackend.eos.compose import (write_compose, load_compose, read_compose,
                                   eos_table_from_arrays, load_simple)
from nmbackend.tov.family import (build_family, default_hc_grid, m_max,
                                  radius_of_mass, lambda_of_mass)
from nmbackend.tov.solver import solve_structure
from nmbackend.reference.scipy_tov import solve_reference


def _rmft_arrays():
    t = RMFTEOS(crust="polytrope").eos_table()
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
    rm = RMFTEOS(crust="polytrope")    # the table reload adds this surface
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


def test_cmf_csv_column_mapping():
    # a MUSES CMF-format CSV (n_B col 5, eps col 8, p col 9, 1-based) must map
    # exactly and give the same TOV as the source EOS.
    import os
    from nmbackend.eos.compose import read_cmf_csv, load_cmf
    n, eps, p = _rmft_arrays()
    M = np.zeros((len(n), 21))
    M[:, 4], M[:, 7], M[:, 8] = n, eps, p              # cols 5, 8, 9
    f = os.path.join(tempfile.mkdtemp(), "CMF_output_stable.csv")
    np.savetxt(f, M, delimiter=",", header="x" * 1, comments="")
    n2, e2, p2 = read_cmf_csv(f)
    assert np.allclose(n2, n) and np.allclose(e2, eps) and np.allclose(p2, p)
    t = load_cmf(f)
    assert np.all(np.diff(np.asarray(t.h)) > 0)


def test_load_simple_columns():
    n, eps, p = _rmft_arrays()
    import os
    f = os.path.join(tempfile.mkdtemp(), "eos.dat")
    np.savetxt(f, np.column_stack([n, eps, p]), header="n_B eps p")
    t = load_simple(f)
    assert np.all(np.diff(np.asarray(t.h)) > 0)
    assert float(t.n[0]) <= n[0] + 1e-12


def test_compose_roundtrip_default_crust():
    """The default (SLy crust) table survives a CompOSE round trip."""
    t0 = RMFTEOS().eos_table()
    d = tempfile.mkdtemp()
    write_compose(d, np.asarray(t0.n), np.asarray(t0.eps_mevfm3),
                  np.asarray(t0.p_mevfm3))
    t1 = load_compose(d)
    f0 = build_family(t0, default_hc_grid(t0, n_models=45))
    f1 = build_family(t1, default_hc_grid(t1, n_models=45))
    assert abs(float(m_max(f0)) - float(m_max(f1))) < 1e-3
    assert abs(float(radius_of_mass(f0, 1.4)) - float(radius_of_mass(f1, 1.4))) < 1e-2
    l0, l1 = float(lambda_of_mass(f0, 1.4)), float(lambda_of_mass(f1, 1.4))
    assert abs(l1 / l0 - 1.0) < 1e-2


# First three rows of the default (C4, mu_S=-50, mu_Q=0) MUSES CMF++ v1.1.0
# CMF_output_stable.csv, verbatim.  The file is intentionally HEADERLESS, with the
# 21-column layout of api/OpenAPI_Specifications_CMF.yaml (CMF_output_all_EoS_variables):
# T, muB, muS, muQ, n_B, n_S, n_Q, eps, p, s, sigma, zeta, delta, omega, phi, rho,
# Phi_order, n_B_without_Phi, n_quark, n_octet, n_decuplet.
_CMF_V110_HEADERLESS_ROWS = """\
0.000000000000E+00,9.220000000000E+02,-5.000000000000E+01,0.000000000000E+00,1.535054382784E-01,0.000000000000E+00,7.675271913922E-02,1.414177246857E+02,1.142894070425E-01,0.000000000000E+00,-6.023042915394E+01,-9.903587230150E+01,2.914407693749E-21,2.090598810863E+01,-8.839180221120E-01,8.727992122163E-21,0.000000000000E+00,1.535054382784E-01,0.000000000000E+00,1.535054382784E-01,0.000000000000E+00
0.000000000000E+00,9.240000000000E+02,-5.000000000000E+01,0.000000000000E+00,1.615162218170E-01,0.000000000000E+00,8.075811090851E-02,1.488115125226E+02,4.294764363490E-01,0.000000000000E+00,-5.902029702850E+01,-9.885120226474E+01,2.948522238618E-24,2.183104638072E+01,-9.988935995847E-01,8.982862577318E-24,0.000000000000E+00,1.615162218170E-01,0.000000000000E+00,1.615162218170E-01,0.000000000000E+00
0.000000000000E+00,9.260000000000E+02,-5.000000000000E+01,0.000000000000E+00,1.686952024138E-01,0.000000000000E+00,8.434760120688E-02,1.554519537969E+02,7.598036382333E-01,0.000000000000E+00,-5.797275890143E+01,-9.869724485034E+01,0.000000000000E+00,2.264678636441E+01,-1.107434193636E+00,0.000000000000E+00,0.000000000000E+00,1.686952024138E-01,0.000000000000E+00,1.686952024138E-01,0.000000000000E+00
"""
# Real CMF++ v1.1.0 default-run (C4, mu_S=-50, mu_Q=0) output directory: NMB_CMF_OUTPUT_DIR,
# else the archived run on sporcsubmit (NOT the volatile build/output scratch, which other
# runs overwrite).  Tests on it skip when absent unless NMB_CMF_REQUIRED is set (then fail).
import os as _os
_CMF_REAL_DIR = _os.environ.get(
    "NMB_CMF_OUTPUT_DIR", _os.path.expanduser("~/q2c_deps/cmf-v1.1.0-runs/C4_muS-50_muQ0"))


def _real_cmf(name):
    import pytest
    f = _os.path.join(_CMF_REAL_DIR, name)
    if not _os.path.exists(f):
        if _os.environ.get("NMB_CMF_REQUIRED"):
            raise RuntimeError(f"NMB_CMF_REQUIRED set but {f} is missing")
        pytest.skip(f"real CMF++ v1.1.0 output {name} not available (set NMB_CMF_OUTPUT_DIR)")
    return f


def _write_cmf_sample(header=None):
    import os
    f = os.path.join(tempfile.mkdtemp(), "CMF_output_stable.csv")
    with open(f, "w") as fh:
        if header is not None:
            fh.write(header + "\n")
        fh.write(_CMF_V110_HEADERLESS_ROWS)
    return f


def test_cmf_csv_headerless_keeps_onset_row():
    # Regression: a real (headerless) CMF++ v1.1.0 table must not lose its first data
    # row -- on the default C4 run that is the mu_B = 922 MeV nuclear-matter onset.
    from nmbackend.eos.compose import read_cmf_csv, _count_header_lines
    f = _write_cmf_sample()
    assert _count_header_lines(f) == 0
    n, eps, p = read_cmf_csv(f)
    assert n.shape == (3,)
    assert np.isclose(n[0], 1.535054382784e-01, rtol=0, atol=1e-15)      # col 5
    assert np.isclose(eps[0], 1.414177246857e+02, rtol=0, atol=1e-9)     # col 8
    assert np.isclose(p[0], 1.142894070425e-01, rtol=0, atol=1e-15)      # col 9
    assert np.allclose(n, [1.535054382784e-01, 1.615162218170e-01, 1.686952024138e-01])
    # the old default (skip_header=1) silently dropped exactly that row
    n_old, _, _ = read_cmf_csv(f, skip_header=1)
    assert n_old.shape == (2,) and np.isclose(n_old[0], 1.615162218170e-01)


def test_cmf_csv_header_autodetect_matches_headerless():
    # A text header row is detected and skipped; result is identical to the headerless
    # read (this is the case the original synthetic-CSV test exercised).
    from nmbackend.eos.compose import read_cmf_csv, _count_header_lines
    hdr = ",".join(["T", "muB", "muS", "muQ", "n_B", "n_S", "n_Q", "eps", "p", "s",
                    "sigma", "zeta", "delta", "omega", "phi", "rho", "Phi_order",
                    "n_B_without_Phi", "n_quark", "n_octet", "n_decuplet"])
    f_h, f_0 = _write_cmf_sample(header=hdr), _write_cmf_sample()
    assert _count_header_lines(f_h) == 1
    a, b = read_cmf_csv(f_h), read_cmf_csv(f_0)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_cmf_csv_real_c4_stable_output():
    # End-to-end on the actual CMF++ v1.1.0 default-run stable table.
    from nmbackend.eos.compose import read_cmf_csv, load_cmf
    f = _real_cmf("CMF_output_stable.csv")
    raw = np.genfromtxt(f, delimiter=",")
    assert raw.shape[1] == 21 and np.all(np.isfinite(raw[0]))       # headerless
    n, eps, p = read_cmf_csv(f)
    assert np.isclose(n.min(), raw[:, 4][raw[:, 4] > 1e-6].min())    # onset kept
    assert np.any(np.isclose(n, raw[0, 4]))                          # mu_B=922 row
    t = load_cmf(f)
    assert np.all(np.diff(np.asarray(t.h)) > 0)
    assert np.all(np.diff(np.asarray(t.p_mevfm3)) > 0)


def test_cmf_csv_real_c4_lepton_output():
    # The file docs/muses-integration.md recommends: 11 columns, and it carries
    # metastable p < 0 points that must be filtered so the table stays monotone.
    from nmbackend.eos.compose import read_cmf_csv, load_cmf
    f = _real_cmf("CMF_output_for_Lepton_baryons.csv")
    raw = np.genfromtxt(f, delimiter=",")
    assert raw.shape[1] == 11 and np.any(raw[:, 8] < 0)
    n, eps, p = read_cmf_csv(f)
    assert np.all(p > 0) and np.any(np.isclose(n, raw[0, 4]))
    t = load_cmf(f)
    assert np.all(np.diff(np.asarray(t.h)) > 0)
    assert np.all(np.diff(np.asarray(t.p_mevfm3)) > 0)


def test_cmf_csv_bom_and_pandas_index_header():
    # Excel/Windows round-trip adds a UTF-8 BOM; a pandas re-export of a headerless
    # frame writes the header "0,1,...,20".  Neither may drop or inject a row.
    import os
    from nmbackend.eos.compose import read_cmf_csv, _count_header_lines
    ref = read_cmf_csv(_write_cmf_sample())
    d = tempfile.mkdtemp()
    f_bom = os.path.join(d, "bom.csv")
    with open(f_bom, "w", encoding="utf-8-sig") as fh:
        fh.write(_CMF_V110_HEADERLESS_ROWS)
    f_idx = _write_cmf_sample(header=",".join(str(i) for i in range(21)))
    assert _count_header_lines(f_bom) == 0 and _count_header_lines(f_idx) == 1
    for f in (f_bom, f_idx):
        for x, y in zip(read_cmf_csv(f), ref):
            assert np.array_equal(x, y)


def test_cmf_csv_rejects_empty_or_nonnumeric():
    import os
    import pytest
    from nmbackend.eos.compose import read_cmf_csv
    d = tempfile.mkdtemp()
    for name, text in [("empty.csv", ""), ("text.csv", "a,b,c\nd,e,f\n"),
                       ("narrow.csv", "1,2,3\n4,5,6\n")]:
        f = os.path.join(d, name)
        open(f, "w").write(text)
        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")        # genfromtxt's empty-file warning
            read_cmf_csv(f)
