"""Matched FD thermodynamics for prescribed-density work; no hydro solver."""
from __future__ import annotations
from pathlib import Path
import sys,math
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'fd_fluid_element'))
import element

def state(T,nB,Ye,*,photons=False):
 s=element.eos(T,nB,Ye,photons=photons);m=s['matter'];n,p,e=m.neutron,m.proton,m.electron
 An=n.mu-T*n.dmu_dT_fixed_density;Ap=p.mu-T*p.dmu_dT_fixed_density;Ae=e.chemical_potential-T*e.dmu_dtemperature_fixed_net_density
 P=m.baryon_pressure+e.pressure;entropy=n.entropy_density+p.entropy_density+e.entropy_density
 PT=entropy+n.density*n.dmu_dT_fixed_density+p.density*p.dmu_dT_fixed_density+e.net_density*e.dmu_dtemperature_fixed_net_density
 if photons:
  eg=math.pi**2*T**4/(15*element.physics.HBARC**3);P+=eg/3;entropy+=4*eg/(3*T);PT+=4*eg/(3*T)
 un=((1-Ye)*An+Ye*(Ap+Ae)-s['u_MeV_per_baryon'])/nB
 return s|dict(P_MeV_fm3=P,entropy_per_baryon=entropy/nB,u_n_MeV_fm3=un,dP_dT_fixed_nY=PT,T=T,nB=nB,Ye=Ye)

def temperature_at_entropy(target,nB,Ye,Tguess,*,photons=False):
 if not math.isfinite(target) or not math.isfinite(Tguess) or Tguess<=0:raise ValueError('Finite entropy and positive Tguess required')
 fn=lambda T:state(T,nB,Ye,photons=photons)['entropy_per_baryon']-target
 lo=Tguess/2;hi=Tguess*2
 for _ in range(20):
  if fn(lo)<=0<=fn(hi):break
  lo/=2;hi*=2
 else:raise ArithmeticError('Entropy inversion failed to bracket')
 return float(brentq(fn,lo,hi,xtol=1e-12,rtol=1e-13))

def adiabatic_move(T,n0,Ye,n1,*,photons=False,independent_work=True):
 before=state(T,n0,Ye,photons=photons);S=before['entropy_per_baryon']
 endT=temperature_at_entropy(S,n1,Ye,T,photons=photons);after=state(endT,n1,Ye,photons=photons)
 work_from_energy=after['u_MeV_per_baryon']-before['u_MeV_per_baryon']
 work=None;error=None
 if independent_work:
  def pressure_path(x):
   n=n0*math.exp(x);temp=temperature_at_entropy(S,n,Ye,T,photons=photons)
   return state(temp,n,Ye,photons=photons)['P_MeV_fm3']/n
  work,error=quad(pressure_path,0,math.log(n1/n0),epsabs=1e-11,epsrel=2e-10)
 return dict(T=endT,nB=n1,Ye=Ye,entropy_before=S,entropy_after=after['entropy_per_baryon'],u_before=before['u_MeV_per_baryon'],u_after=after['u_MeV_per_baryon'],work_energy=work_from_energy,work_pressure_integral=work,work_quad_error=error)

def log_density(theta,amplitude=.02):
 return amplitude*math.sin(math.pi*theta)**2

def density(theta,n0=1e-4,amplitude=.02):return n0*math.exp(log_density(theta,amplitude))
