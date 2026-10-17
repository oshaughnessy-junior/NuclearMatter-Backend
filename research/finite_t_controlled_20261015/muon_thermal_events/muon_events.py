"""Moving free mu- decay/inverse total moments with both neutrino partners.

Unpolarized tree-level GF, no CKM/QED; fixed isotropic FD material bath.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'muon_decay_contract'));import decay as vacuum
sys.path.insert(0,str(R/'relativistic_free_eos'));import free_baryons as eos
M=vacuum.MMU;ME=vacuum.ME;HBARC=eos.HBARC;C=1/(math.pi**2*HBARC**3)
@dataclass(frozen=True)
class Bath:
 T:float
 mu_mu:float
 mu_e:float
 def __post_init__(self):
  if not all(math.isfinite(x)for x in asdict(self).values())or self.T<=0:raise ValueError('Finite material total chemical potentials and T>0')
@dataclass(frozen=True)
class Radiation:
 temperature:float
 mu_anti_e:float
 mu_nu_mu:float
 def __post_init__(self):
  if not all(math.isfinite(x)for x in asdict(self).values())or self.temperature<=0:raise ValueError('Finite radiation chemical potentials,T>0')

def _positive(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='exact_zero')
 if not math.isfinite(log):raise ArithmeticError('Nonfinite total-moment log')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else'underflow')

def parent_kinetic_upper(bath,radiation,blocking,tail):
 """Common-T inverse product is bounded by exp[-(Ep-mu_e-mu_a-mu_n)/T]."""
 center=max(0.,bath.mu_mu-M)
 if radiation is not None and blocking:
  center=max(center,bath.mu_e+radiation.mu_anti_e+radiation.mu_nu_mu-M)
 return center+tail*bath.T

def integrate(bath,radiation=None,*,electron_blocking=True,parent_order=20,pair_order=20,pair_angle=12,orientation_order=8,azimuth_order=12,tail=50.,phase='pair',budget_seconds=110.):
 """Return forward/backward event number and e/antinue/numu energies per volume.

 radiation=None means both partner occupations are exactly zero, not LTE.
 phase='pair' and 'dalitz' use independent rest-frame phase coordinates.
 """
 controls=[parent_order,pair_order,pair_angle,orientation_order,azimuth_order]
 if any(isinstance(v,bool)or not isinstance(v,int)or v<4 for v in controls)or not math.isfinite(tail)or not 30<=tail<=100 or not math.isfinite(budget_seconds)or not 0<budget_seconds<=120 or not isinstance(electron_blocking,bool)or phase not in ['pair','dalitz']:raise ValueError('Finite controls: integer orders>=4,tail30..100,budget<=120,explicit bool and phase')
 if radiation is not None and not isinstance(radiation,Radiation):raise ValueError('Radiation or None required')
 if not isinstance(bath,Bath):raise ValueError('Bath required')
 if radiation is not None and radiation.temperature!=bath.T:raise ValueError('Selected reference requires common-T FD radiation; hotter inverse-tail control remains open')
 started=time.monotonic();x,w=leggauss(pair_order);y,v=leggauss(pair_angle)
 if phase=='pair':
  theta=(x+1)*math.pi/4;s=(M-ME)**2*np.sin(theta)**2;ws=(M-ME)**2*2*np.sin(theta)*np.cos(theta)*w*math.pi/4;S,Y=np.meshgrid(s,y,indexing='ij');pm,e,a,n,p=vacuum.pair_events(S.ravel(),Y.ravel(),M,ME);weight=(ws[:,None]*v[None,:]).ravel()*p/(256*math.pi**3*M*M*vacuum.HBAR)
 else:
  Emax=(M*M+ME*ME)/(2*M);z=(x+1)/2*math.sqrt(Emax-ME);E=ME+z*z;we=w*math.sqrt(Emax-ME)*z;E,Y=np.meshgrid(E,y,indexing='ij');E=E.ravel();p=np.sqrt((E-ME)*(E+ME));lo=(M-E-p)/2;hi=(M-E+p)/2;Ea=(lo+hi)/2+(hi-lo)/2*Y.ravel();pm,e,a,n=vacuum.dalitz_events(E,Ea,M,ME);weight=(we[:,None]*v[None,:]).ravel()*(hi-lo)/2/(64*math.pi**3*M*vacuum.HBAR)
 # Accepted exact invariant amplitude, initial muon spin averaged. Parent
 # radial measure below already has spin2: C=1/(pi² hbarc³), not C/2.
 amp=vacuum.matrix_squared(pm,e,a,n,method='invariant');restweight=weight*amp
 orient,ow=leggauss(orientation_order);phi=2*math.pi*np.arange(azimuth_order)/azimuth_order
 U,P=np.meshgrid(orient,phi,indexing='ij');u=U.ravel();ph=P.ravel();aw=np.repeat(ow/2/azimuth_order,azimuth_order)
 # Rotate event plane about rest z, then tilt its axis relative to parent lab z.
 # Only rotated pz is needed for isotropic bath occupations and energy moments.
 def rotated(p):
  pz=p[:,3,None]*u[None,:]-p[:,1,None]*np.sqrt(1-u*u)[None,:]*np.cos(ph)[None,:]
  return p[:,0,None]+np.zeros_like(pz),pz
 rest=[rotated(p)for p in[e,a,n]];base=(restweight[:,None]*aw[None,:]).ravel();logbase=np.log(base);rest=[(E.ravel(),pz.ravel())for E,pz in rest]
 px,pw=leggauss(parent_order);upper=parent_kinetic_upper(bath,radiation,electron_blocking,tail)
 # x=sqrt(Eparent-M) resolves the radial threshold smoothly.
 cuts=sorted({0.,math.sqrt(upper)}|{math.sqrt(k)for z in [-20,-8,-3,0,3,8,20]if 0<(k:=bath.mu_mu-M+z*bath.T)<upper})
 logsF=[[],[],[],[],[]];logsB=[[],[],[],[],[]];signed=np.zeros(5);event_balance=0.;count=0
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for xx,ww in zip(px,pw):
   if time.monotonic()-started>budget_seconds:raise TimeoutError('Moving-muon event integral exceeded work budget')
   root=lo+(xx+1)*(hi-lo)/2;k=root*root;Ep=M+k;p=math.sqrt(k*(k+2*M));gamma=Ep/M;beta=p/Ep
   # spin2 p²dp, proper/lab time m/E, and sqrt(k) Jacobian.
   parentweight=C*p*Ep*(2*root)*ww*(hi-lo)/2*(M/Ep)
   energies=[gamma*(E+beta*pz)for E,pz in rest];ee,ea,en=energies
   residual=np.max(abs(ee+ea+en-Ep))/Ep;event_balance=max(event_balance,float(residual));count+=len(base)
   lfmu=float(log_expit((bath.mu_mu-Ep)/bath.T));lvmu=float(log_expit((Ep-bath.mu_mu)/bath.T))
   lfe=log_expit((bath.mu_e-ee)/bath.T)if electron_blocking else np.full_like(ee,-np.inf)
   lve=log_expit((ee-bath.mu_e)/bath.T)if electron_blocking else np.zeros_like(ee)
   if radiation is None:
    lfa=np.full_like(ea,-np.inf);lfn=np.full_like(en,-np.inf);lva=np.zeros_like(ea);lvn=np.zeros_like(en)
   else:
    lfa=log_expit((radiation.mu_anti_e-ea)/radiation.temperature);lva=log_expit((ea-radiation.mu_anti_e)/radiation.temperature)
    lfn=log_expit((radiation.mu_nu_mu-en)/radiation.temperature);lvn=log_expit((en-radiation.mu_nu_mu)/radiation.temperature)
   F=lfmu+lve+lva+lvn;B=lvmu+lfe+lfa+lfn
   common=logbase+math.log(parentweight);fullF=common+F;fullB=common+B
   # Stable local signed event accumulation; no enforced F/B ratio or KMS.
   scale=np.maximum(F,B);magnitude=-np.expm1(-np.abs(F-B));delta=np.where(F>=B,magnitude,-magnitude)
   local=base*parentweight*np.exp(scale)*delta
   values=[np.ones_like(ee),ee,ea,en,np.full_like(ee,Ep)]
   for i,value in enumerate(values):
    signed[i]+=float(np.sum(local*value));logsF[i].append(float(logsumexp(fullF+np.log(value))));logsB[i].append(float(logsumexp(fullB+np.log(value))))
 keys=['number','electron_energy','antinue_energy','numu_energy','parent_energy'];forward={};backward={}
 for i,key in enumerate(keys):
  forward[key]=_positive(float(logsumexp(logsF[i])))
  lb=float(logsumexp(logsB[i]));backward[key]=_positive(lb if math.isfinite(lb)else None)
 net=dict(zip(keys,map(float,signed)));Rnet=net['number'];qrad=net['antinue_energy']+net['numu_energy']
 ledgers=dict(material_electron_number=Rnet,material_muon_number=-Rnet,antinue_number=Rnet,numu_number=Rnet,material_energy=-qrad,radiation_energy=qrad,electric_charge_rate=0.,Yp_number_rate=0.)
 entropy=None
 if radiation is not None and electron_blocking:
  sm=(-qrad+(bath.mu_mu-bath.mu_e)*Rnet)/bath.T
  sr=(qrad-(radiation.mu_anti_e+radiation.mu_nu_mu)*Rnet)/radiation.temperature
  entropy=dict(material=sm,radiation=sr,total=sm+sr,units='kB fm^-3 s^-1; prescribed FD baths; no grid entropy claim')
 return dict(forward=forward,backward=backward,net=net,ledgers=ledgers,entropy=entropy,bath=asdict(bath),radiation=asdict(radiation)if radiation else None,controls=dict(parent_order=parent_order,pair_order=pair_order,pair_angle=pair_angle,orientation_order=orientation_order,azimuth_order=azimuth_order,tail=tail,phase=phase,electron_blocking=electron_blocking,parent_kinetic_upper=upper),max_event_energy_residual=event_balance,evaluations=count,runtime_seconds=time.monotonic()-started,units=dict(number='fm^-3 s^-1',energy='MeV fm^-3 s^-1'))
