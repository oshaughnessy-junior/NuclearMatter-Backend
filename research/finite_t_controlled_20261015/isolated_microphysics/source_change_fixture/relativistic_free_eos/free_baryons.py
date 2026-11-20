"""Particle-only ideal relativistic spin2 baryons, MeV/fm units, total mu."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import math,sys
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit,log_expit
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'thermodynamics'));import lepton_reference as leptons
HBARC=leptons.HBARC;C=1/(math.pi**2*HBARC**3)
MN=939.56542052;MP=938.27208816
@dataclass(frozen=True)
class Species:
 temperature:float
 mass:float
 chemical_potential:float
 kinetic_chemical_potential:float
 density:float
 kinetic_energy_density:float
 energy_density:float
 pressure:float
 entropy_density:float
 free_energy_density:float
 cv_fixed_density:float
 dn_dmu_fixed_temperature:float
 dn_dtemperature_fixed_mu:float
 dmu_dtemperature_fixed_density:float
 log_antiparticle_density:float
 antiparticle_density:float
 antiparticle_density_representable:bool
 log_antiparticle_to_particle_ratio:float
 def asdict(self):return asdict(self)

def _controls(T,mu,m,rtol,tail,anti_max_ratio):
 if not all(math.isfinite(x)for x in (T,mu,m,rtol,tail,anti_max_ratio))or T<=0 or m<=0 or not 0<rtol<1e-4 or not 30<=tail<=100 or not 0<anti_max_ratio<1:raise ValueError('Finite T,m>0,rtol(0,1e-4),tail30..100,antiparticle limit(0,1) required')
def _breaks(T,nu,tail):
 end=max(0.,nu)+tail*T
 return sorted(set([0.,end]+[nu+k*T for k in (-60.,-12.,-4.,0.,4.,12.,60.)if 0<nu+k*T<end]))
def _log_density(T,mu,m,rtol,tail):
 # Pull out f(E=m): the remaining scaled occupation is <=1 and integrable.
 scale=float(log_expit((mu-m)/T));es=_breaks(T,mu-m,tail);integral=0.
 def integrand(k):
  z0=(mu-m)/T
  difference=(-k/T+math.log1p(math.exp(z0))-math.log1p(math.exp(z0-k/T))) if z0<=0 else float(log_expit((mu-m-k)/T))-scale
  return math.sqrt(k*(k+2*m))*(k+m)*math.exp(difference)
 for lo,hi in zip(es[:-1],es[1:]):
  val,err,info=quad_vec(integrand,lo,hi,epsabs=1e-24,epsrel=rtol,full_output=True)
  if not info.success and not(info.status==2 and err<=rtol*abs(val)):raise ArithmeticError('Diagnostic density quadrature failed')
  integral+=float(val)
 if integral<=0:raise ArithmeticError('Scaled diagnostic density unresolved')
 return math.log(C)+scale+math.log(integral)

def state_mu(T,mu,mass=MN,*,rtol=2e-12,tail=60.,anti_max_ratio=1e-8):
 """Total-mu forward state; reject appreciable equilibrium antibaryons."""
 T,mu,m=map(float,(T,mu,mass));_controls(T,mu,m,rtol,tail,anti_max_ratio);nu=mu-m
 def integrand(k):
  E=m+k;p=math.sqrt(k*(k+2*m));jac=p*E;x=(k-nu)/T;f=expit(-x);w=f*expit(x);ax=abs(x);s=np.logaddexp(0.,-ax)+ax*expit(-ax)
  return jac*np.array([f,k*f,p*p/(3*E)*f,s,w,(k-nu)*w,(k-nu)**2*w])
 moments=np.zeros(7)
 es=_breaks(T,nu,tail)
 for lo,hi in zip(es[:-1],es[1:]):
  value,error,info=quad_vec(integrand,lo,hi,epsabs=1e-24,epsrel=rtol,norm='max',limit=500,full_output=True)
  if not info.success and not(info.status==2 and error<=max(1e-24,rtol*np.max(abs(value)))):raise ArithmeticError('Baryon energy quadrature failed')
  moments+=value
 n,K,P,s,A0,A1,A2=C*moments
 if not np.all(np.isfinite(moments))or n<=0 or A0<=0:raise ArithmeticError('Particle density/susceptibility not represented')
 variance=A2-A1*A1/A0
 if variance<=0:raise ArithmeticError('Fixed-density covariance not resolved positive')
 loganti=_log_density(T,-mu,m,rtol,tail);logratio=loganti-math.log(n)
 if logratio>math.log(anti_max_ratio):raise ValueError('Equilibrium antibaryon diagnostic exceeds particle-only ansatz limit')
 anti=math.exp(loganti);energy=m*n+K
 return Species(T,m,mu,nu,float(n),float(K),float(energy),float(P),float(s),float(energy-T*s),float(variance/T**2),float(A0/T),float(A1/T**2),float(-A1/(T*A0)),float(loganti),anti,anti>0,float(logratio))

def state_density(T,n,mass=MN,*,rtol=2e-12,tail=60.,anti_max_ratio=1e-8):
 """Positive particle-density inversion; zero baryon species is rejected."""
 T,n,m=map(float,(T,n,mass));_controls(T,m,m,rtol,tail,anti_max_ratio)
 if not math.isfinite(n)or n<=0:raise ValueError('Strictly positive finite particle density; zero species unsupported')
 # Density root itself must not reject a bracket solely for its antiparticle diagnostic.
 def density(nu):return math.exp(_log_density(T,m+nu,m,rtol,tail))
 pf=HBARC*(3*math.pi**2*n)**(1/3);guess=pf*pf/(math.hypot(m,pf)+m);width=max(T,guess,1.)
 lo=guess-width;hi=guess+width
 for _ in range(80):
  if density(lo)<=n<=density(hi):break
  width*=2;lo=guess-width;hi=guess+width
 else:raise ArithmeticError('Positive particle density could not be bracketed')
 nu=brentq(lambda v:density(v)/n-1,lo,hi,xtol=max(5e-13,1e-13*T),rtol=1e-14)
 result=state_mu(T,m+nu,m,rtol=rtol,tail=tail,anti_max_ratio=anti_max_ratio)
 if abs(result.density/n-1)>3e-10:raise ArithmeticError('Density root residual gate failed')
 return result

@dataclass(frozen=True)
class Matter:
 temperature:float
 nB:float
 Ye:float
 Ymu:float
 neutron:Species
 proton:Species
 electron:leptons.State
 muon:leptons.State
 energy_density:float
 pressure:float
 entropy_density:float
 free_energy_density:float
 cv_fixed_composition:float
 zeta_e:float
 zeta_mu:float
 def asdict(self):return asdict(self)

def matter_state(T,nB,Ye,Ymu,*,rtol=2e-12,tail=60.,anti_max_ratio=1e-8):
 """Independent net lepton fractions; physical Yp=Ye+Ymu, no beta equilibrium."""
 if not all(math.isfinite(x)for x in (T,nB,Ye,Ymu))or T<=0 or nB<=0 or not 0<Ye+Ymu<1:raise ValueError('Finite T,nB>0 and0<Ye+Ymu<1 required; zero baryon species rejected')
 Yp=Ye+Ymu;n=state_density(T,nB*(1-Yp),MN,rtol=rtol,tail=tail,anti_max_ratio=anti_max_ratio);p=state_density(T,nB*Yp,MP,rtol=rtol,tail=tail,anti_max_ratio=anti_max_ratio)
 e=leptons.state_density(T,nB*Ye,leptons.ELECTRON_MASS,rtol=rtol,tail=tail);mu=leptons.state_density(T,nB*Ymu,leptons.MUON_MASS,rtol=rtol,tail=tail)
 states=(n,p,e,mu);energy=sum(s.energy_density for s in states);pressure=sum(s.pressure for s in states);entropy=sum(s.entropy_density for s in states);cv=n.cv_fixed_density+p.cv_fixed_density+e.cv_fixed_net_density+mu.cv_fixed_net_density
 return Matter(T,nB,Ye,Ymu,n,p,e,mu,energy,pressure,entropy,energy-T*entropy,cv,e.chemical_potential+p.chemical_potential-n.chemical_potential,mu.chemical_potential+p.chemical_potential-n.chemical_potential)

def temperature_from_energy(U,nB,Ye,Ymu,Tguess,**kwargs):
 """Recover T at fixed nB,Ye,Ymu from total physical energy density."""
 if not all(math.isfinite(x)for x in (U,Tguess))or Tguess<=0:raise ValueError('Finite U and positive guess required')
 fn=lambda T:matter_state(T,nB,Ye,Ymu,**kwargs).energy_density-U
 lo=Tguess/2;hi=Tguess*2
 for _ in range(20):
  if fn(lo)<=0<=fn(hi):break
  if fn(lo)>0:lo/=2
  if fn(hi)<0:hi*=2
 else:raise ArithmeticError('Temperature could not be bracketed inside qualified ansatz')
 return brentq(fn,lo,hi,xtol=2e-10,rtol=2e-12)

def matter_derivatives(matter):
 """Specific-energy derivatives at fixed other(nB,T,Ye,Ymu); all densities net for leptons."""
 m=matter;n,p,e,mu=m.neutron,m.proton,m.electron,m.muon
 An=n.chemical_potential-m.temperature*n.dmu_dtemperature_fixed_density;Ap=p.chemical_potential-m.temperature*p.dmu_dtemperature_fixed_density
 Ae=e.chemical_potential-m.temperature*e.dmu_dtemperature_fixed_net_density;Am=mu.chemical_potential-m.temperature*mu.dmu_dtemperature_fixed_net_density
 u=m.energy_density/m.nB;Yp=m.Ye+m.Ymu
 return dict(u_MeV_per_baryon=u,u_T=m.cv_fixed_composition/m.nB,u_Ye=Ap-An+Ae,u_Ymu=Ap-An+Am,u_nB=((1-Yp)*An+Yp*Ap+m.Ye*Ae+m.Ymu*Am-u)/m.nB)
