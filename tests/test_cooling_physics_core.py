"""JAX core neutrino emissivities (nmbackend.cooling.physics.core_nu)
against NSCool, zone by zone, on NSCool's own stars and Tc.  Skips unless
NSCool is built; NSCOOL_REQUIRED=1 (CI) makes it run."""
import os

import jax
import numpy as np
import pytest

from nmbackend.cooling import default_home

pytestmark = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")

from nmbackend.cooling.physics import core_nu as C  # noqa: E402
from nmbackend.cooling.physics import reference as R  # noqa: E402

TERMS = ("qmurca_nucl", "qbrem_nucl", "qdurca_np", "qpbf_n1s0", "qpbf_n3p2", "qpbf_p1s0")


@jax.jit
def _core(T, z, isf, ca):
    a = dict(kfe=z["kfe"], kfm=z["kfm"], kfn=z["kfn"], kfp=z["kfp"], mstn=z["mstn"],
             mstp=z["mstp"], tcn=z["Tcn"], tcp=z["Tcp"], i=z["i"], isf=isf)
    b = {k: a[k] for k in ("kfn", "kfp", "mstn", "mstp", "tcn", "tcp", "i", "isf")}
    return {"qmurca_nucl": C.murca(T, **a), "qbrem_nucl": C.brem(T, **b),
            "qdurca_np": C.durca(T, nb=z["nb"], Ye=z["Ye"], Ymu=z["Ymu"], **a),
            "qpbf_p1s0": C.pbf_1s0(T, z["Tcp"], z["mstp"], z["kfp"]),
            "qpbf_n1s0": jax.numpy.where(z["i"] > isf, C.pbf_1s0(T, z["Tcn"], z["mstn"], z["kfn"]), 0.0),
            "qpbf_n3p2": jax.numpy.where(z["i"] <= isf, C.pbf_n3p2(T, z["Tcn"], z["mstn"], z["kfn"], ca), 0.0)}


def _case(mass, gaps):
    d = R.dump(mass, gaps)
    core = d.zones[d.zones <= d.icore]
    z = {k: d.star[k][core] for k in ("kfe", "kfm", "kfn", "kfp", "mstn", "mstp", "Tcn",
                                      "Tcp", "nb", "Ye", "Ymu")}
    z["i"] = core
    T = d.terms["T"][:, :core.size] / d.star["ephi"][core]
    return d, core, z, T, _core(T, z, d.isf, R.GAPS[gaps].ca_n3p2)


@pytest.mark.parametrize("mass", [1.4, 2.0])
@pytest.mark.parametrize("gaps", ["none", "default", "zhou", "zhou_no3p2", "c_ccdk"])
def test_core_processes_match_nscool(mass, gaps):
    d, core, z, T, q = _case(mass, gaps)
    total = d.terms["qnu"][:, :core.size]
    for k in TERMS:
        ref = d.terms[k][:, :core.size]
        rel, small = R.gate_error(q[k], ref, total)
        assert rel < 1e-5 and small < 1e-8, (k, rel, small)
    assert np.array_equal(np.asarray(C.durca_flag(z["kfe"], z["kfm"], z["kfn"], z["kfp"])),
                          d.star["idurca_np"][core])


def test_two_gap_durca_suppression_is_exercised():
    """APR 2.0 with gap "c" and CCDK at s_x = 2: dUrca under both p1S0 and n3P2."""
    d, core, z, T, q = _case(2.0, "c_ccdk")
    both = ((d.star["idurca_np"][core] > 0) & (T <= z["Tcn"]) & (T <= z["Tcp"])
            & (core < d.isf))
    assert both.sum() >= 5
    ref = d.terms["qdurca_np"][:, :core.size][both]
    np.testing.assert_allclose(np.asarray(q["qdurca_np"])[both], ref, rtol=1e-5)


def test_ca_scales_n3p2_pair_breaking():
    """a_A = c_A g_A^2: c_A = 2 (2010 release) is 4x the default 0.5."""
    d, core, z, T, _ = _case(1.4, "default")
    q05 = C.pbf_n3p2(T, z["Tcn"], z["mstn"], z["kfn"], 0.5)
    q2 = C.pbf_n3p2(T, z["Tcn"], z["mstn"], z["kfn"], 2.0)
    on = np.asarray(q05) > 0
    assert on.any()
    np.testing.assert_allclose(np.asarray(q2)[on], 4 * np.asarray(q05)[on], rtol=1e-12)
