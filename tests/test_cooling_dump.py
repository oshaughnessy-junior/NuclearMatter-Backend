"""Per-zone dump of NSCool_iso.out (NSCool.zone_dump), the reference of the
JAX physics tests.  Skips unless NSCool is built; NSCOOL_REQUIRED=1 (CI)
makes it run."""
import os

import numpy as np
import pytest

from nmbackend.cooling import GapModel, NSCool, default_home

pytestmark = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")

T_DUMP = [1e7, 1e8, 1e9, 1e10]
NU = ("qeebrem", "qnpb", "qplasma", "qsynch", "qbubble", "qpair", "qphoto",
      "qbrem_nn", "qmurca_nucl", "qbrem_nucl", "qmurca_hyp", "qbrem_hyp",
      "qdurca_np", "qdurca_lap", "qdurca_smn", "qdurca_smla", "qdurca_sms0",
      "qfast", "qdurca_q", "qmurca_q", "qpbf_n1s0", "qpbf_n3p2", "qpbf_p1s0",
      "qpbf_q")
CV = ("cv_n", "cv_p", "cv_e", "cv_m", "cv_l", "cv_sm", "cv_s0", "cv_sp",
      "cv_q", "cv_ion")


@pytest.fixture(scope="module", params=[1.4, 2.0], ids=["APR1.4", "APR2.0"])
def dump(request):
    # the scan grid holds the dump temperatures, so the two can be compared
    return NSCool().zone_dump(T_DUMP, mass=request.param, eta=1e-9,
                              gaps=GapModel(sx_p1s0=0.8, sy_p1s0=0.4),
                              T_grid=(1e7, 1e10, 4))


def test_terms_add_up(dump):
    t = dump.terms
    np.testing.assert_allclose(sum(t[k] for k in NU), t["qnu"], rtol=1e-7)
    np.testing.assert_allclose(sum(t[k] for k in CV), t["cv"], rtol=1e-7)


def test_zone_sums_equal_the_scan(dump):
    """C = sum cv dV and L_nu = sum e^(2 phi) q dV over the dumped zones
    reproduce the scan rows at the same T."""
    st, z, t = dump.star, dump.zones, dump.terms
    np.testing.assert_allclose(dump.iso.T, dump.T, rtol=1e-8)
    dv = st["dvol"][z] + st["dvol"][z + 1]
    w = st["ephi"][z] ** 2 * dv
    np.testing.assert_allclose((t["cv"] * dv).sum(1), dump.iso.C, rtol=1e-7)
    np.testing.assert_allclose((t["qnu"] * w).sum(1), dump.iso.L_nu, rtol=1e-7)
    np.testing.assert_allclose((t["qdurca_np"] * w).sum(1), dump.iso.L_durca,
                               rtol=1e-7, atol=1e-300)
    crust = z > dump.icore
    np.testing.assert_allclose((t["qnu"] * w)[:, crust].sum(1),
                               dump.iso.L_nu_crust, rtol=1e-7)
    mb = t["qmurca_nucl"] + t["qbrem_nucl"] + t["qmurca_hyp"] + t["qbrem_hyp"] + t["qmurca_q"]
    pbf = t["qpbf_n1s0"] + t["qpbf_n3p2"] + t["qpbf_p1s0"] + t["qpbf_q"]
    np.testing.assert_allclose((mb * w).sum(1), dump.iso.L_murca_brem, rtol=1e-7)
    np.testing.assert_allclose((pbf * w).sum(1), dump.iso.L_pbf, rtol=1e-7)


def test_columns_carry_their_physics(dump):
    """Each column is zero where its process is off, so a swapped label fails."""
    st, z, t = dump.star, dump.zones, dump.terms
    core = z <= dump.icore
    T = t["T"] / st["ephi"][z]
    on = lambda k: t[k] != 0
    assert np.array_equal(on("cv_m")[0], st["Ymu"][z] > 0)
    for k in ("cv_ion", "qplasma", "qnpb", "qeebrem"):
        assert not on(k)[:, core].any() and on(k)[:, ~core].any(), k
    for k in ("qmurca_nucl", "qbrem_nucl", "cv_p"):
        assert not on(k)[:, ~core].any() and on(k)[:, core].all(), k
    assert not on("qpbf_n3p2")[:, z > dump.isf].any()
    # pair breaking only below Tc (it underflows to 0 far below)
    n1 = (T <= st["Tcn"][z]) & (z > dump.isf)
    assert on("qpbf_n1s0").any() and not (on("qpbf_n1s0") & ~n1).any()
    p1 = T <= st["Tcp"][z]
    assert not (on("qpbf_p1s0") & ~p1).any()
    assert on("qpbf_p1s0")[(T > 0.5 * st["Tcp"][z]) & p1].all()
    assert not on("qbrem_nn")[:, core | (st["rho"][z] < dump.rhodrip)].any()
    np.testing.assert_allclose(t["fp"], t["lam"] * st["nb"][z] / 3.826e33, rtol=1e-6)


def test_star_arrays(dump):
    st = dump.star
    assert dump.imax % 2 == 1 and st["i"].size == dump.imax + 2
    assert st["rho"][dump.icore] == pytest.approx(dump.rhocore)
    assert np.all(np.diff(st["rho"][:dump.imax + 1]) < 0)     # centre outwards
    assert dump.eta_nscool == pytest.approx(1e-9 / 0.443, rel=1e-6)
    assert dump.qimp == pytest.approx(1e-10)
    core = st["i"] <= dump.icore
    crust = ~core & (st["i"] <= dump.imax)
    assert np.all(st["Tcp"][crust] == 1.0)                    # p1S0 only in the core
    # dUrca flags: 1 (electrons) or 2 (electrons and muons) where the k_F
    # triangle closes; none in a 1.4 Msun APR star
    durca = st["idurca_np"][core] > 0
    assert durca.any() == (st["emas"][dump.imax] > 1.9)
