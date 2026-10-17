"""Selected lepton-integrated GM1 capture, Guo mean-field current ansatz.

Independent log emission/absorption; no crossed beta inventory or rate cache.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import sys,math,time
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'gm1_differential_response'));import adapter as A

def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_sampled_finite_domain_not_global_zero_certificate')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else'underflow')

def integrate_state(state,Enu,ml,mul,*,antineutrino=False,lepton_order=8,angle_order=8,inner_order=16,phi_order=4,inner_tail=45.,outer_tail=24.,coordinate='cos',kernel='gm1',model=A.free.C.FormFactors()):
 if not isinstance(state,A.OrderedState)or not all(math.isfinite(x)for x in(Enu,ml,mul,outer_tail))or Enu<=0 or ml<0 or outer_tail<12:raise ValueError('Valid explicit quasiparticle state, positive incident energy, massive lepton and outer tail required')
 if not isinstance(antineutrino,bool)or coordinate not in('cos','q')or kernel not in('gm1','free_zero_field'):raise ValueError('Explicit sign, phase coordinate and kernel required')
 if any(isinstance(x,bool)or not isinstance(x,int)or x<4 for x in(lepton_order,angle_order)):raise ValueError('Integer outer orders>=4 required')
 if kernel=='free_zero_field'and(state.potential_a!=0 or state.potential_b!=0):raise ValueError('Free anchor requires exactly zero vector fields')
 T=state.T;M=state.dirac_mass;nua=state.total_mu_a-state.potential_a;nub=state.total_mu_b-state.potential_b
 center=Enu+state.potential_a-state.potential_b;p0=math.sqrt(max(0.,center*center-ml*ml));q0=Enu+p0
 velocity=math.sqrt(2*max(T,nua-M,nub-M,0.)/M)
 width=max(T,q0*velocity,q0*q0/(2*M))
 upper=max(ml,center,mul)+outer_tail*T+8*width
 # These are numerical cuts only; exact shifted support is never softened.
 cuts=sorted({ml,upper}|{v for c in(center,mul)for scale in(T,width)for z in(-8,-4,-2,-1,0,1,2,4,8)if ml<(v:=c+z*scale)<upper})
 # Panel exact lepton energies where the shifted light cone crosses either
 # collinear angular endpoint. No support value or rate is altered.
 edges=[];samples=sorted(set(np.linspace(0.,math.sqrt(upper-ml),257))|set(np.sqrt(np.array(cuts)-ml)))
 for sign in(-1,1):
  def boundary(x):
   E=ml+x*x;p=math.sqrt((E-ml)*(E+ml));w=Enu-E-state.potential_b+state.potential_a
   return(Enu+sign*p)**2-w*w
  for a,b in zip(samples[:-1],samples[1:]):
   fa,fb=boundary(a),boundary(b)
   if fa*fb<0:edges.append(ml+brentq(boundary,a,b,xtol=1e-13)**2)
 cuts=sorted(set(cuts+edges))
 xcuts=np.sqrt(np.array(cuts)-ml);xe,we=leggauss(lepton_order);xa,wa=leggauss(angle_order);logs=dict(j=[],chi=[]);empty=0;calls=0;empty_energy_nodes=0;t0=time.monotonic()
 kinetic=A.free.State(T,M,M,nua,nub,state.weak_bare_mass)
 for lo,hi in zip(xcuts[:-1],xcuts[1:]):
  for ex,ew in zip(xe,we):
   y=lo+(ex+1)*(hi-lo)/2;El=ml+y*y;pl=math.sqrt((El-ml)*(El+ml));weight=ew*(hi-lo)*y
   # Equal positive kinetic masses imply |Eb*-Ea*|<|q| for
   # all finite momenta. This partitions exact support, not occupations.
   w=Enu-El-state.potential_b+state.potential_a
   qhi=Enu+pl;qlo=max(abs(Enu-pl),abs(w))
   if qlo>=qhi:empty_energy_nodes+=1;continue
   cmax=min(1.,(Enu*Enu+pl*pl-w*w)/(2*Enu*pl))
   for ax,aw in zip(xa,wa):
    if coordinate=='cos':c=-1+(ax+1)*(cmax+1)/2;angular=aw*(cmax+1)/2
    else:
     q=qlo+(ax+1)*(qhi-qlo)/2;c=(Enu*Enu+pl*pl-q*q)/(2*Enu*pl);angular=aw*(qhi-qlo)/2*q/(Enu*pl)
    if kernel=='gm1':r=A.differential(state,Enu,float(El),float(c),ml,mul,antineutrino=antineutrino,model=model,energy_order=inner_order,phi_order=phi_order,tail=inner_tail)
    else:r=A.free.differential(kinetic,Enu,float(El),float(c),ml,mul,antineutrino=antineutrino,model=model,energy_order=inner_order,phi_order=phi_order,tail=inner_tail)
    calls+=1;empty+=r['chi']['log']is None
    for name in logs:
     if r[name]['log']is not None:logs[name].append(r[name]['log']+math.log(weight*angular))
 result={name:represented(float(logsumexp(values))if values else None)for name,values in logs.items()}
 relaxation=represented(float(logsumexp([result[n]['log']for n in('j','chi')if result[n]['log']is not None]))if any(result[n]['log']is not None for n in('j','chi'))else None)
 affinity=mul+state.total_mu_b-state.total_mu_a
 return dict(**result,relaxation=relaxation,kms_expected=-(Enu-affinity)/T,kms_actual=result['j']['log']-result['chi']['log']if result['j']['log']is not None and result['chi']['log']is not None else None,state=asdict(state),Enu=Enu,lepton_mass=ml,mu_lepton_effective=mul,antineutrino=antineutrino,form_factors=asdict(model),mapping='physical omega=Enu-El; kinetic w=omega-Ub+Ua; physical-q2 form factors and kinetic-qstar WM/PS vertices; common bare939, no extra Q',controls=dict(lepton_order=lepton_order,angle_order=angle_order,inner_order=inner_order,phi_order=phi_order,inner_tail=inner_tail,outer_tail=outer_tail,coordinate=coordinate,kernel=kernel),angular_support='Exact common-positive-mass particle-hole condition q>|w|',panels_El=cuts,upper_El=upper,calls=calls,empty_nodes=empty,empty_energy_nodes=empty_energy_nodes,seconds=time.monotonic()-t0,units='occupation capture j, true absorption chi and relaxation j+chi in s^-1',scope='Capture and inverse capture ONLY; no crossed decay, no RPA, external GM1 forward OPEN')

def integrate(matter,Enu,flavor,antineutrino=False,**controls):
 if flavor not in('e','mu'):raise ValueError('Explicit e or mu flavor required')
 lep=matter.electron if flavor=='e'else matter.muon;ml=A.leptons.ELECTRON_MASS if flavor=='e'else A.leptons.MUON_MASS
 result=integrate_state(matter.ordered(antineutrino),Enu,ml,(-1 if antineutrino else 1)*lep.chemical_potential,antineutrino=antineutrino,**controls)
 result['material_state']=matter.asdict();result['flavor']=flavor
 return result
