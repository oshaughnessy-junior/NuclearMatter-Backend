"""Core-conductivity and crust-boundary measurements on cardassia/SPORC."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import json
import socket

import jax
import jax.numpy as jnp
import numpy as np

from nmbackend.cooling import GapModel
from nmbackend.cooling.physics import reference as R
from nmbackend.cooling.physics.curve import isothermal_curve
from nmbackend.cooling.physics.star import Star, isothermal_sums, zone_terms, crust_march, _l_gamma

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output",type=Path)
a = p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
cases = []
for mass,gaps in ((1.4,"none"),(1.4,"default"),(2.,"zhou"),(2.,"c_ccdk")):
    d = R.dump(mass,gaps,1e-9)
    s = Star.from_dump(d)
    tc = s.tc(R.GAPS[gaps])
    terms = jax.jit(jax.vmap(lambda T: zone_terms(T,s,tc)))(jnp.asarray(d.T))
    sums = jax.jit(jax.vmap(lambda T: isothermal_sums(T,s,tc,d.eta_nscool,rho_b=d.rhocore)))(jnp.asarray(d.T))
    core = d.zones<=d.icore
    error = float(np.max(abs(np.asarray(terms["lam"])[:,core]/d.terms["lam"][:,core]-1)))
    tin_error = float(np.max(abs(np.asarray(sums.T_in)/d.iso.T_in-1)))
    assert error<1e-6 and tin_error<1e-7
    cases.append({"mass":mass,"gaps":gaps,"core_lambda_max_relative_error":error,
                  "boundary_Tin_max_relative_error":tin_error})
    print(json.dumps(cases[-1]),flush=True)

# Pairing-aware boundary, full reverse path including inactive crust formulas.
d = R.dump(1.4,"default",1e-9)
s = Star.from_dump(d)
x = jnp.log(jnp.array([5e8,1e-9,.8]))
def endpoint(x,n_iter):
    T,eta,sy = jnp.exp(x)
    tc = s.tc(GapModel(),sy_p1s0=sy)
    lg,_ = _l_gamma(T,s,eta)
    return jnp.log(crust_march(T,s,lg,s.rhocore,n_iter=n_iter,tc=tc))
values,grads = {},{}
for n in (20,40):
    f = jax.jit(lambda x: endpoint(x,n))
    values[n] = float(f(x))
    grads[n] = np.asarray(jax.jit(jax.grad(lambda x: endpoint(x,n)))(x))
    assert np.all(np.isfinite(grads[n]))
fd = np.array([float((endpoint(x+jnp.eye(3)[i]*1e-5,40)-endpoint(x-jnp.eye(3)[i]*1e-5,40))/2e-5) for i in range(3)])
np.testing.assert_allclose(grads[40],fd,rtol=5e-4,atol=1e-8)
np.testing.assert_allclose(values[20],values[40],atol=1e-9,rtol=0.)
np.testing.assert_allclose(grads[20],grads[40],atol=1e-8,rtol=1e-6)
# A fixed late-time state tests coupled microphysics, not early transport AD.
grid = jnp.logspace(4,10,241)
def cooling_curve(x):
    eta,sy,mass_scale = jnp.exp(x)
    star = dataclasses.replace(s,mstp=s.mstp*mass_scale)
    tc = star.tc(GapModel(),sy_p1s0=sy)
    return isothermal_curve(grid,star,tc,eta,s.rhocore)
x_cooling = jnp.log(jnp.array([1e-9,.8,1.]))
baseline = jax.jit(cooling_curve)(x_cooling)
assert np.all(np.isfinite(baseline.t_yr)) and np.all(np.diff(baseline.t_yr)>0)
anchor_teff = float(jnp.exp(jnp.interp(jnp.log(1e3),jnp.log(baseline.t_yr),jnp.log(baseline.Teff_inf))))
def cooling_endpoint(x):
    c = cooling_curve(x)
    iso_h = jnp.exp(jnp.interp(jnp.log(anchor_teff),jnp.log(c.Teff_inf[::-1]),jnp.log(c.t_yr[::-1])))
    return jnp.interp(jnp.log(iso_h+9e3),jnp.log(c.t_yr),jnp.log(c.Teff_inf))
f_cooling = jax.jit(cooling_endpoint)
g_cooling = np.asarray(jax.jit(jax.grad(cooling_endpoint))(x_cooling))
assert np.all(np.isfinite(g_cooling))
fd_cooling = np.array([float((f_cooling(x_cooling+jnp.eye(3)[i]*1e-4)-f_cooling(x_cooling-jnp.eye(3)[i]*1e-4))/2e-4) for i in range(3)])
np.testing.assert_allclose(g_cooling,fd_cooling,rtol=5e-4,atol=1e-7)
# Gate every bracket between handoff and endpoint for baseline and FD cases.
for dx in (jnp.zeros(3),*[sign*jnp.eye(3)[i]*1e-4 for i in range(3) for sign in (-1,1)]):
    c = cooling_curve(x_cooling+dx)
    assert c.Teff_inf[-1]<=anchor_teff<=c.Teff_inf[0]
    iso_h = np.exp(np.interp(np.log(anchor_teff),np.log(c.Teff_inf[::-1]),np.log(c.t_yr[::-1])))
    ages = np.asarray(c.t_yr)
    assert ages[0]<=iso_h and iso_h+9e3<=ages[-1]
    lo = max(0,np.searchsorted(ages,iso_h)-1)
    hi = min(len(ages),np.searchsorted(ages,iso_h+9e3)+1)
    assert np.all((c.T[lo:hi]>=grid.min())&(c.T[lo:hi]<=grid.max()))
cooling = {"parameter_order":["log_eta","log_proton_gap_amplitude","log_proton_Landau_mass_scale"],
           "fixed_1kyr_anchor_Teff":anchor_teff,"10kyr_Teff":float(np.exp(f_cooling(x_cooling))),
           "logTeff_gradient":g_cooling.tolist(),"finite_difference":fd_cooling.tolist(),
           "max_absolute_AD_FD_error":float(np.max(abs(g_cooling-fd_cooling)))}
result = {"host":socket.gethostname(),"base_commit":"f97b5fd","cases":cases,
          "paired_boundary_logTin_20":values[20],"paired_boundary_logTin_40":values[40],
          "paired_boundary_gradient_20":grads[20].tolist(),
          "paired_boundary_gradient_40":grads[40].tolist(),
          "paired_boundary_finite_difference":fd.tolist(),
          "coupled_late_cooling":cooling,
          "scope":"NSCool2009 ICON_CORE2 lepton-only; fixed stellar background; transition derivatives excluded"}
a.output.joinpath("results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2),flush=True)
