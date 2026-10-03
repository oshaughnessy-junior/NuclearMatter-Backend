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
            **dict(zip(("qpbf_n1s0", "qpbf_n3p2"),
                       C.pbf_neutron(T, z["Tcn"], z["mstn"], z["kfn"], z["i"], isf, ca)))}


def _case(mass, gaps):
    d = R.dump(mass, gaps)
    core = d.zones[d.zones <= d.icore]
    z = {k: d.star[k][core] for k in ("kfe", "kfm", "kfn", "kfp", "mstn", "mstp", "Tcn",
                                      "Tcp", "nb", "Ye", "Ymu")}
    z["i"] = core
    T = d.terms["T"][:, :core.size] / d.star["ephi"][core]
    return d, core, z, T, _core(T, z, d.isf, R.GAPS[gaps].ca_n3p2)


@pytest.mark.parametrize("mass", [1.4, 2.0])
@pytest.mark.parametrize("gaps", ["none", "default", "zhou", "zhou_no3p2", "c_ccdk",
                                  "ccdk_n", "n_only", "p_ccdk"])
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


def test_branch_coverage_on_our_stars():
    """The cases above reach the proton-only dUrca branch and an unpaired
    proton at zone isf (where only the neutron factor acts)."""
    d, core, z, T, _ = _case(2.0, "p_ccdk")
    du = d.star["idurca_np"][core] > 0
    assert (du & (T > z["Tcn"]) & (T <= z["Tcp"])).sum() >= 5
    d, core, z, T, _ = _case(1.4, "n_only")
    k = list(core).index(d.isf)
    assert ((T[:, k] < z["Tcn"][k]) & (T[:, k] >= z["Tcp"][k])).sum() >= 3


def test_isf_conventions():
    """neutrino_core.f: zone isf is 1S0 in Urca (i >= isf), 3P2 in pair breaking."""
    T, tc = 5e8, 1e9
    a = dict(kfe=0.5, kfm=0.3, kfn=2.0, kfp=0.6, mstn=0.8, mstp=0.7, tcn=tc, tcp=1.0)
    at = lambda i, isf: float(C.murca(T, i=i, isf=isf, **a))
    assert at(5, 5) == at(5, 0) != at(5, 6)
    q1, q3 = C.pbf_neutron(T, tc, 0.8, 2.0, 5, 5)
    assert float(q1) == 0.0 and float(q3) > 0.0


def test_unreached_reductions_match_nscool_routines(nscool_probe):
    """func_r_1s0_1s0 (both 1S0; no zone of ours), func_r_1s0_3P2B including
    its radius > 3 branch, and pair breaking, against NSCool's own routines."""
    rng = np.random.default_rng(3)
    v = rng.uniform(0, 40, (60, 2)) * rng.choice([0, 1, 1], (60, 2))
    t = 10 ** rng.uniform(-6, 0, (80, 2))
    pb = np.column_stack([10 ** rng.uniform(8, 10, 40), rng.uniform(0.5, 1, 40),
                          rng.uniform(0.3, 2.5, 40)])
    pb = np.column_stack([pb[:, 0] * 10 ** rng.uniform(-2, 0.02, 40), pb])
    ref = nscool_probe([("r11", a, b, 0, 0) for a, b in v]
                       + [("r13", a, b, 0, 0) for a, b in t]
                       + [(k, *r) for r in pb for k in ("pbf1", "pbf3")])
    tab = C.sf_suppression_table()
    got = np.concatenate([
        np.asarray(C._r_1s0_1s0(v[:, 0], v[:, 1])),
        np.asarray(C._r_1s0_3p2b(t[:, 0], t[:, 1], tab)),
        np.ravel([[float(C.pbf_1s0(T, tc, m, k)), float(C.pbf_n3p2(T, tc, m, k))]
                  for T, tc, m, k in pb])])
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-300)
    assert (np.hypot(*np.log10(t).T) > 3).any() and (np.hypot(*np.log10(t).T) < 3).any()


def test_muon_durca_and_gradients():
    """flag 2 adds the muon term; d/d(nb, Ymu, Ye) finite where Ymu = 0."""
    a = dict(kfe=0.5, kfn=1.1, kfp=0.75, mstn=0.8, mstp=0.7, nb=0.6, Ye=0.08,
             tcn=1.0, tcp=1.0, i=0, isf=-1)
    e = float(C.durca(1e9, kfm=0.0, Ymu=0.0, **a))
    m = float(C.durca(1e9, kfm=0.4, Ymu=0.03, **a))
    assert C.durca_flag(0.5, 0.4, 1.1, 0.75) == 2 and C.durca_flag(0.5, 0.3, 1.1, 0.75) == 1
    assert m / e == pytest.approx(1 + (0.03 / 0.08) ** (1 / 3), rel=1e-12)
    g = jax.grad(lambda nb, ymu, ye: C.durca(1e9, kfm=0.0, Ymu=ymu, nb=nb, Ye=ye,
                                             **{k: v for k, v in a.items()
                                                if k not in ("nb", "Ye")}),
                 argnums=(0, 1, 2))(0.6, 0.0, 0.08)
    assert np.all(np.isfinite(g))
