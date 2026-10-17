"""Free relativistic neutron beta/inverse3body spectral coefficients, one antinu.

Research reference. Shared frozen full-current/tensor primitives; no medium.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,math,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_tensor'));import tensor as t
sys.path.insert(0,str(ROOT/'relativistic_free_eos'));import free_baryons as eos
V=t.V;MN=eos.MN;MP=eos.MP;ME=eos.leptons.ELECTRON_MASS;MW=(MN+MP)/2
@dataclass(frozen=True)
class Bath:
 T:float
 mu_n:float
 mu_p:float
 mu_e:float
 def __post_init__(self):
  if not all(math.isfinite(x)for x in asdict(self).values())or self.T<=0:raise ValueError('Finite total chemical potentials and T>0 required')
 @classmethod
 def from_matter(cls,m):
  if (m.neutron.mass,m.proton.mass,m.electron.mass)!=(MN,MP,ME):raise ValueError('Matched physical free masses required')
  return cls(m.temperature,m.neutron.chemical_potential,m.proton.chemical_potential,m.electron.chemical_potential)

def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_support')
 if not math.isfinite(log):raise ArithmeticError('Nonfinite log coefficient')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else'underflow')

def differential(bath,Enu,Ee,c,*,energy_order=12,phi_order=4,tail=50.,model=t.C.FormFactors()):
 if not all(math.isfinite(x)for x in(Enu,Ee,c))or Enu<=0 or Ee<=ME or not -1<=c<=1:raise ValueError('Finite Enu>0,Ee>me,c in[-1,1] required')
 # Validate quadrature controls even when invariant support is empty.
 if isinstance(energy_order,bool)or not isinstance(energy_order,int)or energy_order<8 or isinstance(phi_order,bool)or not isinstance(phi_order,int)or phi_order<4 or not math.isfinite(tail)or not 30<=tail<=120:raise ValueError('Energy>=8,phi>=4 integer controls; tail30..120')
 pe=math.sqrt((Ee-ME)*(Ee+ME));q2=(Enu-pe)**2+2*Enu*pe*(1+c);q=math.sqrt(q2);omega=Enu+Ee
 invariant=ME*ME+2*Enu*(Ee-pe*c)
 if invariant>(MN-MP)**2 or q==0:
  if q==0 and invariant<=(MN-MP)**2:raise ValueError('Exact q=0 coordinate requires separate limit')
  return dict(j=represented(None),chi=represented(None),invariant=invariant,support=[],units='s^-1 MeV^-1 per dcos')
 state=t.State(bath.T,MP,MN,bath.mu_p,bath.mu_n,MW)
 r=t.response(state,omega,q,energy_order=energy_order,phi_order=phi_order,tail=tail,model=model)
 kz=(q2+Enu*Enu-pe*pe)/(2*q);kx2=Enu*Enu-kz*kz
 if kx2< -1e-10*Enu*Enu:raise ArithmeticError('Crossed lepton q-frame shell error')
 k=np.array([Enu,math.sqrt(max(0.,kx2)),0.,kz]);l=np.array([Ee,-k[1],0.,q-kz]);left=V.G@(V.I-V.G5)
 # inverse lepton current vbar_anti Gamma u_e, independent of positron capture
 L=np.array([[np.trace(V.slash(k)@left[u]@(V.slash(l)+ME*V.I)@V.adjoint(left[v]))for v in range(4)]for u in range(4)])
 pref=V.GF**2*V.VUD**2*pe/(32*math.pi**2*Enu*t.HBAR)
 chi=t._coefficient(r['absorption'],L,log_expit((bath.mu_e-Ee)/bath.T),pref)
 j=t._coefficient(r['emission'],L,log_expit((Ee-bath.mu_e)/bath.T),pref)
 return dict(j=j,chi=chi,invariant=invariant,support=r['support'],response=r,lepton_tensor=L,k=k,l=l,units='s^-1 MeV^-1 per dcos')

def spectral(bath,Enu,*,angle_order=8,lepton_order=12,widths=9.,energy_order=12,phi_order=4,tail=50.,model=t.C.FormFactors(),budget_seconds=110.):
 """Integrate selected sub-rest-endpoint energy with Doppler-resolved Ee panels.

 Ee truncation is empirical/refinement-qualified, not a universal tail bound.
 Differential API has full invariant support including moving endpoint tails.
 """
 if not .01<=bath.T<=.1 or max((bath.mu_n-MN)/bath.T,(bath.mu_p-MP)/bath.T)>-8:raise ValueError('Selected integrated rule requires T .01.. .1 and dilute baryon eta<=-8; no broad spectral domain certificate')
 if not math.isfinite(Enu)or not 0<Enu<((MN-MP-ME)*(MN+MP+ME))/(2*MN):raise ValueError('Selected integrated rule restricted below rest antinu endpoint')
 if any(isinstance(v,bool)or not isinstance(v,int)or v<4 for v in(angle_order,lepton_order))or not math.isfinite(widths)or not 4<=widths<=16 or not math.isfinite(budget_seconds)or not 0<budget_seconds<=120:raise ValueError('Finite bounded quadrature controls required')
 for v,lo in[(energy_order,8),(phi_order,4)]:
  if isinstance(v,bool)or not isinstance(v,int)or v<lo:raise ValueError('Internal quadrature controls required')
 if not math.isfinite(tail)or not 30<=tail<=120:raise ValueError('Tail30..120')
 x,w=leggauss(angle_order);y,v=leggauss(lepton_order);logs=[[],[]];evaluations=0;start=time.monotonic();panels=[]
 for c,wc in zip(x,w):
  def shell(E):
   p=math.sqrt(max(0.,(E-ME)*(E+ME)));return E+Enu+math.sqrt(MP*MP+p*p+Enu*Enu+2*p*Enu*c)-MN
  center=brentq(shell,ME,MN-MP,xtol=2e-13);p=math.sqrt((center-ME)*(center+ME));q=math.sqrt(Enu*Enu+p*p+2*Enu*p*c);width=q*math.sqrt(bath.T/MN)
  edges=sorted({max(ME,center-widths*width),center,max(ME,center-3*width),center+3*width,center+widths*width})
  panels.append(dict(c=float(c),center=center,width=width,lower=edges[0],upper=edges[-1]))
  for lo,hi in zip(edges[:-1],edges[1:]):
   for xx,ww in zip(y,v):
    if time.monotonic()-start>budget_seconds:raise TimeoutError('Selected spectrum exceeded bounded runtime; no result qualified')
    E=lo+(xx+1)*(hi-lo)/2;weight=wc*ww*(hi-lo)/2
    d=differential(bath,Enu,E,float(c),energy_order=energy_order,phi_order=phi_order,tail=tail,model=model);evaluations+=1
    for i,name in enumerate(('j','chi')):
     if d[name]['log']is not None:logs[i].append(math.log(weight)+d[name]['log'])
 return dict(j=represented(float(logsumexp(logs[0]))if logs[0]else None),chi=represented(float(logsumexp(logs[1]))if logs[1]else None),Enu=Enu,bath=asdict(bath),controls=dict(angle_order=angle_order,lepton_order=lepton_order,widths=widths,energy_order=energy_order,phi_order=phi_order,tail=tail),panels=panels,evaluations=evaluations,runtime_seconds=time.monotonic()-start,units='j,chi s^-1 per one antinu occupation; DOS E²/(2pi² hbarc³) fm^-3 MeV^-1')

def matched_npe(T,nB,Ye):
 """Neutral free npe bath; muons absent, avoiding an unrepresented muon EOS."""
 if not all(math.isfinite(x)for x in(T,nB,Ye))or T<=0 or nB<=0 or not 0<Ye<1:raise ValueError('Positive finite T,nB and interior Ye required')
 n=eos.state_density(T,nB*(1-Ye),MN);p=eos.state_density(T,nB*Ye,MP);e=eos.leptons.state_density(T,nB*Ye,ME)
 return Bath(T,n.chemical_potential,p.chemical_potential,e.chemical_potential),dict(T=T,nB=nB,Ye=Ye,neutron=n.asdict(),proton=p.asdict(),electron=asdict(e),muons='absent')
