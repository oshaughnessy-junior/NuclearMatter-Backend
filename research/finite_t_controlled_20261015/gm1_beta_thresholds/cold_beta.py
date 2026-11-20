"""Cold common939GM1 beta-equilibrated e/mu matter and Fermi triangles."""
from pathlib import Path
from dataclasses import dataclass,asdict
import math,sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'rmf_reference'));import gm1_finite_t as gm
sys.path.insert(0,str(ROOT/'thermodynamics'));import lepton_reference as lepton
H=gm.HBARC;ME=lepton.ELECTRON_MASS;MMU=lepton.MUON_MASS

def lepton_cold(mu,m):
 if not all(math.isfinite(x)for x in(mu,m))or m<=0 or mu<0:raise ValueError('Finite mu>=0 and m>0 required')
 if mu<=m:return dict(mu=mu,mass=m,kF_MeV=0.,density=0.,energy=0.,pressure=0.,susceptibility=0.)
 p=math.sqrt((mu-m)*(mu+m));n=p**3/(3*math.pi**2*H**3)
 # Direct scalar momentum integrals avoid cold analytic cancellation at p/m<<1.
 E=quad(lambda k:k*k*math.hypot(k,m),0,p,epsabs=1e-18,epsrel=2e-12)[0]/(math.pi**2*H**3)
 P=quad(lambda k:k**4/math.hypot(k,m),0,p,epsabs=1e-18,epsrel=2e-12)[0]/(3*math.pi**2*H**3)
 return dict(mu=mu,mass=m,kF_MeV=p,density=n,energy=E,pressure=P,susceptibility=p*mu/(math.pi**2*H**3))

def cold_hessian(nuclear):
 """Exact T0 canonical nuclear Hessian from cold Fermi-surface responses."""
 m=nuclear.dirac_mass_mev/H;phi=nuclear.phi_mev/H;M=gm.GM1['M_mev']/H;dens=[nuclear.nB*(1-nuclear.Yp),nuclear.nB*nuclear.Yp];a=[];diagonal=[];scalar=0.
 for n in dens:
  k=gm.kf(n);E=math.hypot(k,m);a.append(m/E);diagonal.append(math.pi**2/(k*E));scalar+=quad(lambda p:p**4/(p*p+m*m)**1.5,0,k,epsabs=1e-14,epsrel=2e-12)[0]/math.pi**2
 D=1/gm.GM1['C_sigma']+2*gm.GM1['b']*M*phi+3*gm.GM1['c']*phi*phi+scalar
 co=gm.GM1['C_omega'];cr=gm.GM1['C_rho']/4;h=(np.diag(diagonal)-np.outer(a,a)/D+np.array([[co+cr,co-cr],[co-cr,co+cr]]))*H
 return h,D
@dataclass(frozen=True)
class BetaState:
 nB:float
 Yp:float
 Ye:float
 Ymu:float
 mu_lepton:float
 electron:dict
 muon:dict
 nuclear:gm.NuclearState
 nuclear_hessian:tuple
 charge_relaxed_hessian:tuple
 nuclear_eigenvalues:tuple
 charge_relaxed_eigenvalues:tuple
 scalar_curvature:float
 energy_density:float
 pressure:float
 delta_e_MeV:float
 delta_mu_MeV:float
 neutrality_residual:float
 beta_residual_MeV:float
 def asdict(self):return asdict(self)

def state(nB,*,electron_mass=ME):
 if not math.isfinite(nB)or nB<=0 or not math.isfinite(electron_mass)or electron_mass<=0:raise ValueError('Finite nB,m_e>0 required')
 def residual(Yp):
  s=gm.cold_state(nB,Yp);mu=s.mu_n_mev-s.mu_p_mev
  if mu<=0:return -nB*Yp
  return lepton_cold(mu,electron_mass)['density']+lepton_cold(mu,MMU)['density']-nB*Yp
 lo,hi=1e-6,.499999
 if not residual(lo)>0>residual(hi):raise ArithmeticError('Beta/neutrality root not bracketed')
 Yp=brentq(residual,lo,hi,xtol=2e-14,rtol=2e-14);n=gm.cold_state(nB,Yp);mu=n.mu_n_mev-n.mu_p_mev;e=lepton_cold(mu,electron_mass);m=lepton_cold(mu,MMU);ne,nmu=e['density'],m['density'];h,D=cold_hessian(n);hcharged=h.copy();hcharged[1,1]+=1/(e['susceptibility']+m['susceptibility']);ev=np.linalg.eigvalsh(h);ec=np.linalg.eigvalsh(hcharged)
 charge=(ne+nmu)/nB-Yp
 beta_res=abs(mu-math.hypot(H*gm.kf(ne),electron_mass))
 if nmu>0:beta_res=max(beta_res,abs(mu-math.hypot(H*gm.kf(nmu),MMU)))
 if abs(charge)>2e-11 or beta_res>1e-10 or n.max_normalized_equation_residual>2e-10 or D<=0:raise ArithmeticError('Cold beta forward residual/scalar gate failed')
 if ec[0]<=0:raise ValueError('Charge-relaxed uniform branch locally unstable')
 kn=H*gm.kf(nB*(1-Yp));kp=H*gm.kf(nB*Yp)
 return BetaState(nB,Yp,ne/nB,nmu/nB,mu,e,m,n,tuple(map(tuple,h.tolist())),tuple(map(tuple,hcharged.tolist())),tuple(ev.tolist()),tuple(ec.tolist()),D,n.energy_density+e['energy']+m['energy'],n.pressure+e['pressure']+m['pressure'],kn-kp-e['kF_MeV'],kn-kp-m['kF_MeV'],charge,beta_res)

def threshold(flavor,lo,hi,**kwargs):
 if flavor not in('e','mu'):raise ValueError('e or mu threshold required')
 key='delta_e_MeV'if flavor=='e'else'delta_mu_MeV';fun=lambda density:getattr(state(density,**kwargs),key)
 if fun(lo)*fun(hi)>=0:raise ValueError('Actual Fermi-triangle sign bracket required')
 root=brentq(fun,lo,hi,xtol=2e-12,rtol=2e-12);return state(root,**kwargs)
