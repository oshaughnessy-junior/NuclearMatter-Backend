"""Static S(q)=1 screened Born-Mott electron Lorentz gas, natural MeV units."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,math
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'relativistic_free_eos'))
import free_baryons as F
ALPHA=1/137.035999084;E2=4*math.pi*ALPHA;ME=.51099895069;HC=197.3269804
@dataclass(frozen=True)
class State:
 T:float;ne_fm3:float;Z:int;mu:float;screening_squared:float;ni_natural:float;screening_susceptibility_natural:float;pair_log_ratio:float;pair_ratio_representable:bool;pair_ratio:float;gas:dict
 def asdict(self):return asdict(self)
def state(T,ne,Z):
 if not all(math.isfinite(x)for x in(T,ne))or min(T,ne)<=0 or isinstance(Z,bool)or not isinstance(Z,int)or Z<=0:raise ValueError('Finite T,ne>0 and positive integer ion Z required')
 # Generic relativistic spin-two particle EOS, with ELECTRON mass; no baryon convention imported.
 s=F.state_density(T,ne,ME);chi=s.dn_dmu_fixed_temperature*HC**3;ratio=math.exp(s.log_antiparticle_to_particle_ratio)
 return State(T,ne,Z,s.chemical_potential,E2*chi,ne*HC**3/Z,chi,s.log_antiparticle_to_particle_ratio,ratio>0,ratio,s.asdict())
def controls(p,ks2,Z):
 if not all(math.isfinite(x)for x in(p,ks2,Z))or min(p,ks2,Z)<=0:raise ValueError('Finite strictly positive p,screening squared,Z required')
def dcs(p,c,ks2,Z):
 controls(p,ks2,Z)
 if not math.isfinite(c)or not -1<=c<=1:raise ValueError('Finite physical scattering cosine required')
 x=(1-c)/2;return 4*Z*Z*ALPHA*ALPHA*(ME*ME+p*p*(1-x))/(4*p*p*x+ks2)**2

def coulomb_integral(a,beta2):
 """I=int0^1 dx x(1-beta2*x)/(x+a)^2; no fitted logarithm."""
 if not math.isfinite(a)or a<=0 or not math.isfinite(beta2)or not 0<=beta2<=1:raise ValueError('Finite a>0 and0<=v²<=1')
 if a>=4:
  u=1/a;J=math.fsum((-u)**k*(k+1)*(1/(k+2)-beta2/(k+3))for k in range(100));I=J*u*u
 else:
  L=math.log1p(1/a);I=(1+2*a*beta2)*L-1/(1+a)-beta2*(1+a/(1+a))
 if not math.isfinite(I)or I<=0:raise ArithmeticError('Positive Coulomb integral unrepresented')
 return I

def frequency(p,s):
 if not isinstance(s,State):raise ValueError('Explicit screening state required')
 controls(p,s.screening_squared,s.Z);E=math.hypot(p,ME);v=p/E;u=4*p*p/s.screening_squared
 if u<=.25:
  # Algebraically identical strong-screening series, retaining p²/v=pE before tiny-p cancellation.
  J=math.fsum((-u)**k*(k+1)*(1/(k+2)-v*v/(k+3))for k in range(100));nu=32*math.pi*s.ni_natural*s.Z**2*ALPHA**2*(p*E)/(s.screening_squared**2)*J
 else:nu=2*math.pi*s.ni_natural*s.Z**2*ALPHA**2/(p*p*v)*coulomb_integral(1/u,v*v)
 if not math.isfinite(nu)or nu<=0:raise ArithmeticError('Positive collision frequency unrepresented')
 return nu

def moments(s,*,tail=60.,rtol=2e-10):
 if not isinstance(s,State)or not math.isfinite(tail)or not 25<=tail<=100 or not math.isfinite(rtol)or not 0<rtol<1e-5:raise ValueError('Explicit state, finite tail25..100 andrtol(0,1e-5) required')
 lower=max((ME-s.mu)/s.T,-tail);upper=max(0.,(ME-s.mu)/s.T)+tail
 cuts=sorted(set([lower,upper]+[x for x in[-16.,-6.,0.,6.,16.]if lower<x<upper]));vals=np.zeros(3);errors=0.
 def f(y):
  E=s.mu+s.T*y
  if E<=ME:return np.zeros(3)
  p=math.sqrt((E-ME)*(E+ME));tau=1/frequency(p,s);weight=p**3/E*tau*expit(-y)*expit(y)/(3*math.pi**2)
  return weight*np.array([1.,y,y*y])
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  v,e,info=quad_vec(f,lo,hi,epsabs=1e-20,epsrel=rtol,full_output=True)
  if not info.success:raise ArithmeticError('Energy-window moment quadrature failed')
  vals+=v;errors+=e
 K=vals*np.array([1,s.T,s.T*s.T]);variance=K[2]-K[1]*K[1]/K[0]
 if not np.all(np.isfinite(K))or K[0]<=0 or variance<=0:raise ArithmeticError('Positive thermoelectric covariance unresolved')
 sigma=E2*K[0];kappa=variance/s.T;seebeck=-K[1]/(math.sqrt(E2)*s.T*K[0]);pF=HC*(3*math.pi**2*s.ne_fm3)**(1/3);EF=math.hypot(ME,pF);drude=E2*s.ne_fm3*HC**3/(EF*frequency(pF,s))
 return dict(K=K.tolist(),sigma=float(sigma),kappa_open=float(kappa),kappa_Estar_zero=float(K[2]/s.T),seebeck=float(seebeck),peltier=float(s.T*seebeck),Lorenz_ratio=float(kappa/(sigma*s.T)),WF_limit=math.pi**2/(3*E2),sigma_Fermi_surface=drude,born_Zalpha_over_vF=s.Z*ALPHA/(pF/EF),controls=dict(tail=tail,rtol=rtol,energy_window=[lower,upper]),estimated_scaled_error=errors,units=dict(K0='MeV',K1='MeV²',K2='MeV³',sigma='MeV, Heaviside-Lorentz charge convention',kappa='MeV²',seebeck='dimensionless kB=1, eHL positive'),state=s.asdict())
