"""JAX crust neutrino emissivities and heat capacities (nmbackend.cooling.physics
crust_nu, heat) against NSCool, zone by zone.  Skips unless NSCool is built;
NSCOOL_REQUIRED=1 (CI) makes it run."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.cooling import default_home

pytestmark = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")

from nmbackend.cooling.physics import crust_nu as N  # noqa: E402
from nmbackend.cooling.physics import heat as H  # noqa: E402
from nmbackend.cooling.physics import reference as R  # noqa: E402


@jax.jit
def _terms(T, z, isf, icore, rhodrip):
    i = z["i"]
    core, crust = i <= icore, i > icore
    inner = crust & (z["rho"] >= rhodrip)
    A = jnp.where(crust, z["A_cell"], 1.0)
    Z = jnp.where(crust, z["Z_ion"], 1.0)
    kfe = jnp.where(crust, z["kfe"], 1.0)
    cn = H.cv_coeff(z["kfn"], 939.56 * z["mstn"])
    cp = H.cv_coeff(z["kfp"], 938.27 * z["mstp"])
    return {
        "qeebrem": jnp.where(crust, N.eebrem(T, kfe), 0.0),
        "qnpb": jnp.where(crust, N.npb(T, z["rho"]), 0.0),
        "qplasma": jnp.where(crust, N.plasma(T, z["rho"], A, Z), 0.0),
        "qbrem_nn": jnp.where(inner, N.brem_nn_crust(T, z["kfn"], z["mstn"], z["Tcn"],
                                                     z["v_ion"], i, isf), 0.0),
        "cv_e": H.lepton_cv_coeff(z["kfe"], 0.511) * T,
        "cv_m": H.lepton_cv_coeff(z["kfm"], 105.0) * T,
        "cv_n": jnp.where(z["rho"] >= rhodrip, H.cv_neutron(T, cn, z["Tcn"], i, isf), 0.0),
        "cv_p": jnp.where(core, H.cv_proton(T, cp, z["Tcp"]), 0.0),
        "cv_ion": jnp.where(crust, H.cv_ion(T, z["rho"], A, Z), 0.0),
    }


@pytest.mark.parametrize("mass", [1.4, 2.0])
@pytest.mark.parametrize("gaps", ["none", "default", "zhou", "c_ccdk"])
def test_crust_nu_and_heat_capacity_match_nscool(mass, gaps):
    d = R.dump(mass, gaps)
    z = {k: jnp.asarray(d.star[k][d.zones]) for k in
         ("kfe", "kfm", "kfn", "kfp", "mstn", "mstp", "Tcn", "Tcp", "rho", "A_cell",
          "Z_ion", "v_ion")}
    z["i"] = jnp.asarray(d.zones)
    T = d.terms["T"] / d.star["ephi"][d.zones]
    out = _terms(T, z, d.isf, d.icore, d.rhodrip)
    for k, v in out.items():
        total = d.terms["qnu" if k.startswith("q") else "cv"]
        rel, small = R.gate_error(v, d.terms[k], total)
        assert rel < 1e-5 and small < 1e-8, (k, rel, small)
        assert np.asarray(v).any() or not d.terms[k].any(), k


def test_proton_cv_ramp():
    """NSCool's proton C: reduced value just below Tc, ramp to normal at 1.1 Tc."""
    tc, c = 1e9, 1.0
    below, at, mid, above = (float(H.cv_proton(x * tc, c, tc)) / (x * tc)
                             for x in (0.999999, 1.000001, 1.05, 1.1001))
    assert below == pytest.approx(at, rel=1e-3) and below == pytest.approx(2.426, rel=1e-3)
    assert mid == pytest.approx(0.5 * (below + 1.0), rel=1e-4) and above == 1.0


def test_routines_match_nscool_on_all_branches(nscool_probe):
    """cvion (gas, ramp, liquid, blend, harmonic, Debye), plasma, npb and
    eebrem against NSCool's own routines on random points."""
    rng = np.random.default_rng(5)
    n = 400
    T = 10 ** rng.uniform(5.5, 10.5, n)
    rho = 10 ** rng.uniform(4, 14.2, n)
    A = rng.uniform(12, 3000, n)
    Z = rng.uniform(6, 50, n)
    mu = 10 ** rng.uniform(-0.3, 2.3, n)
    rows = ([("cvion", *r) for r in zip(T, rho, A, Z)]
            + [("plasma", *r) for r in zip(T, rho, A, Z)]
            + [("npb", t, r, 0, 0) for t, r in zip(T, rho)]
            + [("eebrem", t, m, 0, 0) for t, m in zip(T, mu)])
    ref = nscool_probe(rows).reshape(4, n)
    got = np.stack([np.asarray(H.cv_ion(T, rho, A, Z)), np.asarray(N.plasma(T, rho, A, Z)),
                    np.asarray(N.npb(T, rho)), np.asarray(N.eebrem(T, mu / 197.0))])
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-300)
    gamma = 2.273e5 * Z ** 2 * (rho / A) ** (1 / 3) / T
    a1 = np.where(rho >= 4.3e11, 3 * Z, A)
    delta = Z / T * np.sqrt(rho / (a1 * A)) * 6.022e23
    hit = [gamma <= .1, (gamma > .1) & (gamma <= .2), (gamma > .2) & (gamma <= 178),
           (gamma > 178) & (gamma <= 210) & (delta >= 1e19) & (delta < 7e20),
           (gamma > 178) & (delta <= 1e19), (gamma > 210) & (delta > 1e19) & (delta < 7e20),
           (gamma > 210) & (delta >= 7e20)]
    assert all(h.sum() >= 2 for h in hit), [int(h.sum()) for h in hit]


def test_masked_inputs_keep_gradients_finite():
    """Core zones have A = Z = 0 and no Tc; a plain where must not give NaN."""
    f = lambda T: (jnp.where(False, H.cv_ion(T, 1e15, 0.0, 0.0), 0.0)
                   + jnp.where(False, N.plasma(T, 1e15, 0.0, 0.0), 0.0)
                   + jnp.where(False, N.eebrem(T, 0.0), 0.0)
                   + H.cv_proton(T, 1.0, 0.0) + H.cv_neutron(T, 1.0, 0.0, 0, -1)
                   + N.brem_nn_crust(T, 0.5, 0.9, 0.0, 0.1, 0, -1))
    assert np.isfinite(jax.grad(f)(1e8))
    assert float(H.cv_ion(1e8, 1e15, 0.0, 0.0)) == 0.0


def test_crust_nn_brem_triplet_branch():
    """nubrem_crust_nn: 3P2 gap for i < isf, 1S0 for i >= isf (neutrino_core.f)."""
    q = lambda i, isf: float(N.brem_nn_crust(3e8, 1.2, 0.9, 1e9, 0.1, i, isf))
    assert q(5, 6) != q(5, 5) == q(5, 0)
