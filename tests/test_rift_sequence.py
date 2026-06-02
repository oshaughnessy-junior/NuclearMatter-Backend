"""RIFT EOS-sequence I/O: robust NSSequence + legacy EOSSequenceLandry drop-in."""
import os
import tempfile
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from nmbackend.eos import MetaModelEOS, HybridEOS
from nmbackend.rift import write_nss, NSSequence, write_landry_legacy
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
