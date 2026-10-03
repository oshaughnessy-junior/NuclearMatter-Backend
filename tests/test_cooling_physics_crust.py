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
