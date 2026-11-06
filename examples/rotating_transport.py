"""RNS interior experiment: conditional transport, no envelope observables.

Run only on cardassia/SPORC. Inputs exported by export_rns.py. Dimensionless
K=Cv=1 are ansatz coefficients, not inferred nuclear microphysics.
"""
from __future__ import annotations
import argparse
import json
import socket
from pathlib import Path
import time
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from nmbackend.thermal.diffusion import build_operator, metric_factors, implicit_step

parser=argparse.ArgumentParser()
parser.add_argument("csv",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
data=np.genfromtxt(args.csv,delimiter=",",names=True,skip_header=1)
r_native=np.unique(data["r"])
mu_native=np.unique(data["mu"])
assert np.array_equal(data["r"],np.repeat(r_native,len(mu_native)))
assert np.array_equal(data["mu"],np.tile(mu_native,len(r_native)))
fields={key:data[key].reshape(len(r_native),len(mu_native)) for key in data.dtype.names[2:]}
# Find the first vacuum radial node at ANY latitude; truncate well inside it.
outer=np.flatnonzero(np.any(fields["energy"]<=0,axis=1))[0]
rmax=.7*r_native[outer-1]
r_faces=np.linspace(0,rmax,25)
mu_faces=np.linspace(0,1,17)
r=(r_faces[1:]+r_faces[:-1])/2
mu=(mu_faces[1:]+mu_faces[:-1])/2
rr,mm=np.meshgrid(r,mu,indexing="ij")
points=np.stack([rr,mm],axis=-1)
sampled={key:RegularGridInterpolator((r_native,mu_native),v)(points) for key,v in fields.items()}
assert np.all(sampled["energy"]>0)
# Independent metric/export convention check at native grid points in matter.
import re
Omega=float(re.search(r"Omega=([^ ]+)",args.csv.read_text().splitlines()[0])[1])
native_r,native_mu=np.meshgrid(r_native,mu_native,indexing="ij")
predicted=(Omega-fields["omega"])**2*native_r**2*(1-native_mu**2)*np.exp(-2*fields["rho"])
in_matter=fields["energy"]>0
velocity_error=float(np.max(np.abs(predicted[in_matter]-fields["velocity_sq"][in_matter])))
assert velocity_error<1e-4
assert np.all((sampled["velocity_sq"]>=0)&(sampled["velocity_sq"]<1))
N,A,B,G=metric_factors(*(jnp.asarray(sampled[k]) for k in ("rho","gama","alpha","velocity_sq")))
def operator(log_k):
    return build_operator(r_faces,mu_faces,N,A,B,G,1.,jnp.exp(log_k))
op=operator(0.)
u0=1+.2*(rr/rmax)**2*jnp.exp(-((rr/rmax-.6)/.2)**2)*(3*mm**2-1)/2
constant,residual=implicit_step(op,jnp.ones_like(u0),.01*rmax**2)
assert float(residual)<1e-10
# Endpoint variance is nonconserved; fixed geometry and one transport knob.
dt=.005*rmax**2
steps=30
def evolve(log_k):
    op=operator(log_k)
    def advance(u,_):
        v,res=implicit_step(op,u,dt)
        return v,res
    return jax.lax.scan(advance,u0,None,length=steps)
def variance(log_k):
    v,_=evolve(log_k)
    mean=jnp.sum(op.capacity*v)/jnp.sum(op.capacity)
    return jnp.sum(op.capacity*(v-mean)**2)/jnp.sum(op.capacity)
start=time.perf_counter()
v,residuals=evolve(0.)
v.block_until_ready()
forward_seconds=time.perf_counter()-start
ad=float(jax.grad(variance)(0.))
fd={str(h):float((variance(h)-variance(-h))/(2*h)) for h in (1e-2,1e-3,1e-4)}
summary={"host":socket.gethostname(),"domain":"closed spherical inner core; equatorial symmetry",
    "coefficients":"dimensionless Cv=K=1; constant ansatz",
    "shape":list(u0.shape),"rmax":rmax,"input_metadata":args.csv.read_text().splitlines()[0],
    "equilibrium_error":float(jnp.max(jnp.abs(constant-1))),
    "balance_relative":float(abs(jnp.sum(op.capacity*(v-u0))/jnp.sum(op.capacity*u0))),
    "max_solve_residual":float(jnp.max(residuals)),"variance_final":float(variance(0.)),
    "dvariance_dlogK":ad,"finite_differences":fd,
    "gradient_relative_error_h1e-4":abs(ad-fd["0.0001"])/abs(ad),
    "forward_seconds_includes_first_compile":forward_seconds,
    "rns_velocity_absolute_error":velocity_error,"max_lorentz":float(jnp.max(G)),"local_equilibrium_T_range":
    [float(jnp.min(G/N)),float(jnp.max(G/N))]}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(summary,indent=2)+"\n")
np.savez(args.output.with_suffix(".npz"),r=rr,mu=mm,u0=np.asarray(u0),u=np.asarray(v),
         lapse=np.asarray(N),lorentz=np.asarray(G),energy=sampled["energy"])
print(json.dumps(summary,indent=2))
assert summary["max_solve_residual"]<1e-10
assert summary["balance_relative"]<1e-9
assert summary["gradient_relative_error_h1e-4"]<2e-5
