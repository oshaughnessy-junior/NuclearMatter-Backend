"""RIFT EOS-sequence I/O: robust NSSequence + legacy EOSSequenceLandry drop-in."""
import json
import os
import tempfile
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from nmbackend.eos import MetaModelEOS, HybridEOS
from nmbackend.rift import write_nss, NSSequence, write_landry_legacy
from nmbackend.rift.sequence import NSS_FIELDS, NSS_UNITS, _stable_branches
from nmbackend.tov.family import build_family, default_hc_grid, lambda_of_mass, m_max


def _tables():
    mm = MetaModelEOS()
    return [mm.eos_table(), mm.eos_table({"L_sym": 40.0})]


def test_nss_roundtrip_and_metadata():
    tables = _tables()
    params = [{"L_sym": 60}, {"L_sym": 40}]
    fn = write_nss(os.path.join(tempfile.mkdtemp(), "s.h5"), tables, params, n_points=70)
    s = NSSequence(fn)
    assert s.n_eos == 2 and s.params[0]["L_sym"] == 60
    assert "schema" not in s.fields and "M" in s.fields and "hc" in s.fields
    for e in range(2):
        fam = build_family(tables[e], default_hc_grid(tables[e], n_models=70))
        assert abs(s.lambda_of_m(1.4, e) - float(lambda_of_mass(fam, 1.4))) \
            < 0.03 * float(lambda_of_mass(fam, 1.4))
        assert abs(s.m_max(e) - float(m_max(fam))) < 1e-3


def test_legacy_schema_and_consumer_replay():
    tables = _tables()
    fn = write_landry_legacy(os.path.join(tempfile.mkdtemp(), "L.h5"),
                             tables, eos_tables=tables, n_points=120)
    with h5py.File(fn, "r") as f:
        assert "id" in f and "ns" in f and "eos" in f
        name = sorted(f["ns"].keys())[0]
        assert set(np.array(f["ns"][name]).dtype.names) == {"M", "R", "Lambda"}
        assert set(np.array(f["eos"][name]).dtype.names) == \
            {"pressurec2", "energy_densityc2", "baryon_density"}
        # replay EOSSequenceLandry.lambda_of_m_indx (interp log-Lambda vs sorted M)
        dat = np.array(f["ns"][name])
        i = np.argsort(dat["M"])
        lam = float(np.exp(np.interp(1.4, dat["M"][i], np.log(dat["Lambda"][i]))))
        eos = np.array(f["eos"][name])
    fam = build_family(tables[0], default_hc_grid(tables[0], n_models=70))
    assert abs(lam - float(lambda_of_mass(fam, 1.4))) < 0.03 * float(lambda_of_mass(fam, 1.4))
    # baryon rest-mass density at saturation ~ 2.7e14 g/cm^3
    rho_sat = eos["baryon_density"][np.argmin(np.abs(eos["baryon_density"] - 2.7e14))]
    assert 2.0e14 < rho_sat < 3.5e14


def test_legacy_stored_branch_is_single_valued_for_hybrid():
    # a twin/phase-transition EOS: the stored ns branch must be monotone in M (so
    # RIFT's single-valued M->Lambda interp is correct by construction)
    hy = HybridEOS(p_trans_mev=80.0, delta_eps_mev=250.0, cs2_quark=1.0)
    fn = write_landry_legacy(os.path.join(tempfile.mkdtemp(), "H.h5"),
                             [hy.eos_table()], n_points=120)
    with h5py.File(fn, "r") as f:
        dat = np.array(f["ns"][sorted(f["ns"].keys())[0]])
    assert np.all(np.diff(np.sort(dat["M"])) >= 0)
    assert np.all(np.diff(dat["M"]) > -1e-9)        # stored already M-increasing


def test_nss_primary_branch_does_not_mix_disconnected_stable_branches(tmp_path):
    rows = np.full((1, 5, len(NSS_FIELDS)), np.nan)
    col = {name: i for i, name in enumerate(NSS_FIELDS)}
    rows[0, :, col["hc"]] = np.arange(1, 6) / 10
    rows[0, :, col["M"]] = [1.0, 2.0, 1.8, 1.6, 2.1]
    rows[0, :, col["R"]] = [13.0, 11.0, 10.8, 10.0, 9.0]
    rows[0, :, col["Lambda"]] = [600.0, 100.0, 80.0, 60.0, 20.0]
    rows[0, :, col["stable"]] = [1.0, 1.0, 0.0, 1.0, 1.0]
    fn = tmp_path / "twins.nss.h5"
    with h5py.File(fn, "w") as f:
        f.attrs["representation"] = "tabular_hc/1"
        f.attrs["schema_version"] = "nmbackend.nss/1"
        f.attrs["fields"] = json.dumps(NSS_FIELDS)
        f.attrs["units"] = json.dumps(NSS_UNITS)
        f.create_dataset("sequence", data=rows)

    seq = NSSequence(fn)
    assert seq.n_stable_branches(0) == 2
    assert seq.stable_branch_counts.tolist() == [2]
    assert seq.m_max(0) == pytest.approx(2.0)
    assert seq.radius_of_m(1.8, 0) == pytest.approx(11.4)
    expected_lambda = np.exp(np.interp(1.8, [1.0, 2.0], np.log([600.0, 100.0])))
    assert seq.lambda_of_m(1.8, 0) == pytest.approx(expected_lambda)

    branches = _stable_branches(
        rows[0, :, col["M"]], rows[0, :, col["R"]],
        rows[0, :, col["Lambda"]], rows[0, :, col["stable"]]
    )
    assert [branch[0].tolist() for branch in branches] == [[1.0, 2.0], [1.6, 2.1]]


def test_sequence_stable_from_mass_curve_not_ad():
    """Stability comes from M(h_c) on the grid: a row with a wrong AD
    derivative cannot split the branch, and the extended configurations below
    the minimum-mass star and the rows past M_max are not stable."""
    from nmbackend.rift.sequence import _sequence_stable
    hc = np.linspace(0.02, 0.6, 30)
    M = np.r_[0.65, 0.4, 0.2, np.linspace(0.25, 2.1, 22), [2.08, 2.0, 1.9, 1.8, 1.7]]
    R = np.r_[1800.0, 300.0, 40.0, np.full(27, 12.0)]
    Lam = np.full(30, 500.0)
    st = _sequence_stable(hc, M, R, Lam)
    assert not st[:3].any()
    assert st[3:25].all()
    assert not st[25:].any()


def test_one_eos_starts_at_minimum_mass_star():
    """Every stored row lies on compact configurations (no R ~ 10^3 km row),
    and the stable rows form one branch up to M_max."""
    from nmbackend.rift.sequence import _one_eos
    t = MetaModelEOS().eos_table()
    rows = _one_eos(t, n_points=24)
    col = {k: i for i, k in enumerate(NSS_FIELDS)}
    R, M, st = rows[:, col["R"]], rows[:, col["M"]], rows[:, col["stable"]]
    assert np.nanmax(R) < 30.0
    assert len(_stable_branches(M, R, rows[:, col["Lambda"]], st)) == 1
    assert M[st > 0.5].max() == pytest.approx(np.nanmax(M), rel=1e-12)
