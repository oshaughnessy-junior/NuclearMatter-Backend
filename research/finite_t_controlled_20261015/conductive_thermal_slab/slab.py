"""Conservative electron heat slab with actual screened Lorentz conductivity.

Dimensionless x/L,t/(Cv0 L²/kappa0),T/T0; prescribed local-FD closure.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import sys,math,hashlib,platform,time
import numpy as np
from scipy.optimize import brentq
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'screened_coulomb_transport'));import lorentz as L
@dataclass
class Physics:
 T0:float=.041
 ne:float=5.3e-8
 Z:int=2
 tail:float=60.
 rtol:float=2e-10
 def __post_init__(self):
  self.sources=[R/'screened_coulomb_transport/lorentz.py',R/'relativistic_free_eos/free_baryons.py',R/'thermodynamics/lepton_reference.py',Path(__file__)]
  self.fingerprint=self._fingerprint();self.thermo_cache={};self.transport_cache={};self.eos_calls=0;self.transport_calls=0
  ref=self.thermo(self.T0);self.u0=ref.gas['energy_density'];self.s0=ref.gas['entropy_density'];self.cv0=ref.gas['cv_fixed_density'];self.kappa0=self.transport(self.T0)['kappa_open']
 def _fingerprint(self):
  return tuple((str(p),hashlib.sha256(p.read_bytes()).hexdigest())for p in self.sources)+(('python',platform.python_version()),('numpy',np.__version__),('scipy',__import__('scipy').__version__),('inputs',(float(self.T0).hex(),float(self.ne).hex(),self.Z,float(self.tail).hex(),float(self.rtol).hex())))
 def _key(self,T):
  if not math.isfinite(T)or T<=0:raise ValueError('Positive finite electron temperature required')
  if self._fingerprint()!=self.fingerprint:raise RuntimeError('Source/environment/controls changed; exact-state cache invalidated')
  return float(T).hex()
 def thermo(self,T):
  key=self._key(T)
  if key not in self.thermo_cache:self.thermo_cache[key]=L.state(T,self.ne,self.Z);self.eos_calls+=1
  return self.thermo_cache[key]
 def transport(self,T):
  key=self._key(T)
  if key not in self.transport_cache:self.transport_cache[key]=L.moments(self.thermo(T),tail=self.tail,rtol=self.rtol);self.transport_calls+=1
  return self.transport_cache[key]
 def W(self,theta):return (self.thermo(self.T0*float(theta)).gas['energy_density']-self.u0)/(self.cv0*self.T0)
 def entropy(self,theta):return (self.thermo(self.T0*float(theta)).gas['entropy_density']-self.s0)/self.cv0
 def cv(self,theta):return self.thermo(self.T0*float(theta)).gas['cv_fixed_density']/self.cv0
 def k(self,theta):return self.transport(self.T0*float(theta))['kappa_open']/self.kappa0
 def inverse(self,W,guess,bracket):
  lo,hi=bracket;t=guess
  for _ in range(8):
   residual=self.W(t)-W
   if abs(residual)<2e-12:return t
   if residual>0:hi=t
   else:lo=t
   trial=t-residual/self.cv(t)
   t=trial if lo<trial<hi else (lo+hi)/2
  t=brentq(lambda theta:self.W(theta)-W,lo,hi,xtol=5e-13,rtol=5e-14)
  if abs(self.W(t)-W)>5e-11:raise ArithmeticError('Conservative EOS inversion failed')
  return t
 def scales(self,length_m=1e-4):
  if not math.isfinite(length_m)or length_m<=0:raise ValueError('Positive finite slab length required')
  hbar=6.582119569e-22;hc_m=L.HC*1e-15;c=299792458.
  length_natural=length_m/hc_m;cv_natural=self.cv0*L.HC**3
  time_natural=cv_natural*length_natural**2/self.kappa0
  pF=L.HC*(3*math.pi**2*self.ne)**(1/3);EF=math.hypot(pF,L.ME);nu=L.frequency(pF,self.thermo(self.T0));v=pF/EF
  return dict(length_m=length_m,time_unit_seconds=time_natural*hbar,Cv_natural_MeV3=cv_natural,kappa_natural_MeV2=self.kappa0,diffusivity_m2_per_s=self.kappa0/cv_natural*hc_m**2/hbar,screening_length_m=hc_m/math.sqrt(self.thermo(self.T0).screening_squared),pF_mfp_natural=pF*v/nu,mfp_Fermi_m=v/nu*hc_m,Knudsen_Fermi=v/nu*hc_m/length_m,collision_time_Fermi_seconds=hbar/nu,hbar_c_consistency_relative=hbar*c/hc_m-1)

def evolve(physics,theta0,duration,steps,*,reservoirs=None,budget_seconds=110.):
 theta=np.asarray(theta0,dtype=float).copy()
 if theta.ndim!=1:raise ValueError('One-dimensional temperatures required')
 N=len(theta)
 if theta.ndim!=1 or N<4 or not np.all(np.isfinite(theta))or np.min(theta)<=0 or not math.isfinite(duration)or duration<=0 or isinstance(steps,bool)or not isinstance(steps,int)or steps<1 or not math.isfinite(budget_seconds)or not 0<budget_seconds<=120:raise ValueError('Finite positive state/time, N>=4 and bounded steps/runtime')
 if reservoirs is not None and(len(reservoirs)!=2 or not all(math.isfinite(x)and x>0 for x in reservoirs)):raise ValueError('Two positive finite reservoir temperatures or None')
 dx=1/N;dt=duration/steps;alltemps=list(theta)+(list(reservoirs)if reservoirs else[]);bounds=(min(alltemps)*.999,max(alltemps)*1.001);W=np.array([physics.W(t)for t in theta]);initial_energy=float(np.sum(W)*dx);initial_entropy=float(sum(physics.entropy(t)for t in theta)*dx)
 incoming=outgoing=boundary_entropy=production=0.;history=[];start=time.monotonic();inversion_residual=0.
 def harmonic(a,b):return 2*a*b/(a+b)
 for step in range(steps):
  if time.monotonic()-start>budget_seconds:raise TimeoutError('Bounded nonlinear heat experiment stopped')
  conductivity=np.array([physics.k(t)for t in theta]);cv=np.array([physics.cv(t)for t in theta]);stability=dt/dx**2*np.max(conductivity)/np.min(cv)
  if stability>.22:raise ValueError('Explicit nonlinear diffusion CFL exceeds controlled .22 gate')
  q=np.zeros(N+1);q[1:-1]=-(2*conductivity[:-1]*conductivity[1:]/(conductivity[:-1]+conductivity[1:]))*np.diff(theta)/dx
  if reservoirs:
   q[0]=-harmonic(physics.k(reservoirs[0]),conductivity[0])*(theta[0]-reservoirs[0])/(dx/2)
   q[-1]=-harmonic(conductivity[-1],physics.k(reservoirs[1]))*(reservoirs[1]-theta[-1])/(dx/2)
  old=theta.copy();Wnext=W+dt*(q[:-1]-q[1:])/dx
  theta=np.array([physics.inverse(value,t,bounds)for value,t in zip(Wnext,theta)])
  inversion_residual=max(inversion_residual,max(abs(physics.W(t)-w)for t,w in zip(theta,Wnext)));W=Wnext
  incoming+=dt*q[0];outgoing+=dt*q[-1]
  prod=float(np.sum(q[1:-1]*(1/old[1:]-1/old[:-1])))
  if reservoirs:
   prod+=q[0]*(1/old[0]-1/reservoirs[0])+q[-1]*(1/reservoirs[1]-1/old[-1])
   boundary_entropy+=dt*(q[0]/reservoirs[0]-q[-1]/reservoirs[1])
  production+=dt*prod
  physical_energy=float(sum(physics.W(t)for t in theta)*dx);S=float(sum(physics.entropy(t)for t in theta)*dx)
  history.append(dict(time=(step+1)*dt,material_energy=physical_energy,entropy=S,boundary_energy_in=incoming,boundary_energy_out=outgoing,boundary_entropy=boundary_entropy,face_entropy_production_integral=production,max_CFL=stability))
 return dict(theta=theta.tolist(),initial_theta=np.asarray(theta0).tolist(),initial_energy=initial_energy,initial_entropy=initial_entropy,history=history,reservoirs=reservoirs,controls=dict(N=N,duration=duration,steps=steps,dx=dx,dt=dt),max_inversion_residual=inversion_residual,runtime_seconds=time.monotonic()-start,EOS_calls=physics.eos_calls,transport_calls=physics.transport_calls)
