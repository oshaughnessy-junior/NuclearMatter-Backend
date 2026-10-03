"""Core transport: compiled NSCool values, assembled star and physical AD."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.cooling import default_home
from nmbackend.cooling.physics import reference as R
from nmbackend.cooling.physics.core_conduct import lambda_core
from nmbackend.cooling.physics.star import Star, isothermal_sums, zone_terms

reference_required = pytest.mark.skipif(
    not (default_home()/"Code/NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"), reason="NSCool not built")


@reference_required
def test_scalar_prescription_and_absent_species(tmp_path):
    """Separate scalar routine covers no muons/protons/leptons and T <,=,> Tc."""
    code = default_home()/"Code"
    fc = os.environ.get("FC", "gfortran")
    if shutil.which(fc) is None:
        pytest.fail("Fortran compiler required for scalar reference")
    exe = tmp_path/"probe"
    subprocess.run([fc,"-std=legacy","-w","-o",str(exe),
                    str(Path(__file__).with_name("nscool_core_conduct_probe.f")),
                    str(code/"conductivity_core.o")],check=True)
    cases = np.array([[T,ke,km,kp,.8,tc] for T in (1e6,1e8,1e9,1e10)
                      for ke,km,kp,tc in ((1.,.5,1.1,1e9),(1.,0.,1.,1e9),
                                         (1.,0.,0.,0.),(0.,0.,0.,0.))])
    text = "".join(" ".join(map(repr,row.tolist()))+"\n" for row in cases)
    output = subprocess.run([str(exe)],input=text,capture_output=True,text=True,check=True)
    ref = np.array([float(x) for x in output.stdout.split()])
    got = jax.jit(lambda_core)(*cases.T)
    np.testing.assert_allclose(got,ref,rtol=1e-7,atol=0.)
    absent = lambda lp: jnp.log(lambda_core(jnp.exp(lp[0]),1.,0.,0.,jnp.exp(lp[1]),0.))
    assert np.all(np.isfinite(jax.grad(absent)(jnp.log(jnp.array([1e8,.8])))))


@reference_required
@pytest.mark.parametrize("mass,gaps", [(1.4,"none"),(1.4,"default"),
                                       (2.,"zhou"),(2.,"c_ccdk")])
def test_assembled_conductivity_and_boundary_match_nscool(mass,gaps):
    d = R.dump(mass,gaps,1e-9)
    s = Star.from_dump(d)
    tc = s.tc(R.GAPS[gaps])
    r = jax.jit(jax.vmap(lambda t: zone_terms(t,s,tc)))(jnp.asarray(d.T))
    np.testing.assert_allclose(r["lam"],d.terms["lam"],rtol=1e-6)
    sums = jax.jit(jax.vmap(lambda t: isothermal_sums(t,s,tc,d.eta_nscool,
                                                   rho_b=d.rhocore)))(jnp.asarray(d.T))
    np.testing.assert_allclose(sums.T_in,d.iso.T_in,rtol=1e-7)


def test_transport_gradients_include_pairing_and_mass():
    """Forward/reverse AD and FD away from pairing and species thresholds."""
    for km,T in ((.5,1e8),(0.,1e8),(.5,2e9)):
        def f(x):
            temp,mp,tcp = jnp.exp(x)
            return jnp.log(lambda_core(temp,1.,km,1.1,mp,tcp))
        x = jnp.log(jnp.array([T,.8,1e9]))
        forward = jax.jacfwd(f)(x)
        reverse = jax.jit(jax.grad(f))(x)
        fd = jnp.array([(f(x+jnp.eye(3)[i]*1e-5)-f(x-jnp.eye(3)[i]*1e-5))/2e-5
                        for i in range(3)])
        np.testing.assert_allclose(reverse,forward,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(reverse,fd,rtol=2e-5,atol=1e-8)
