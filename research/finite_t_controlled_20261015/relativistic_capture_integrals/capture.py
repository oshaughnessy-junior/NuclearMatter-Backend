"""Lepton-integrated FREE relativistic capture/inverse coefficients.

No crossed decay; no interactions, photons, antinucleons or rate cache.
"""
from __future__ import annotations
from pathlib import Path
import sys,math,time
from dataclasses import asdict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_tensor'));import tensor as K
sys.path.insert(0,str(ROOT/'relativistic_free_eos'));import free_baryons as EOS

def represent(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_support')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else'underflow')

def integrate(matter,Enu,flavor='mu',antineutrino=False,*,lepton_order=8,angle_order=8,energy_order=16,phi_order=4,tail=40.,outer_tail=24.,model=K.C.FormFactors()):
 if flavor not in('e','mu')or not isinstance(antineutrino,bool):raise ValueError('Explicit e/mu and neutrino sign required')
 if any(isinstance(v,bool)or not isinstance(v,int)or v<4 for v in(lepton_order,angle_order)):raise ValueError('Positive integer quadrature orders>=4 required')
 if not math.isfinite(Enu)or Enu<=0 or not math.isfinite(outer_tail)or outer_tail<12:raise ValueError('Positive Enu and outer tail>=12 required')
 a,b=(matter.proton,matter.neutron)if antineutrino else(matter.neutron,matter.proton)
 lep=matter.electron if flavor=='e'else matter.muon
 mass=K.V.ME if flavor=='e'else K.V.MMU
 mul=(-1 if antineutrino else 1)*lep.chemical_potential
 state=K.State(matter.temperature,a.mass,b.mass,a.chemical_potential,b.chemical_potential)
 center=Enu+a.mass-b.mass;p0=math.sqrt(max(0.,center*center-mass*mass));q0=Enu+p0
 # Widths choose numerical panels only. Actual support and occupations are
 # always evaluated by the exact relativistic kernel; no smoothing enters it.
 thermal=q0*math.sqrt(matter.temperature/min(a.mass,b.mass))
 deg=q0*math.sqrt(2*max(0.,a.kinetic_chemical_potential,b.kinetic_chemical_potential)/min(a.mass,b.mass))
 width=max(matter.temperature,thermal,deg,q0*q0/(2*min(a.mass,b.mass)))
 upper=max(mass,center,mul)+outer_tail*matter.temperature+8*width
 cuts=sorted({mass,upper}|{x for c in(center,mul)for scale in(matter.temperature,width)for z in(-8,-4,-2,-1,0,1,2,4,8)if mass<(x:=c+z*scale)<upper})
 # El=m+x² resolves the exact massive-lepton endpoint.
 cuts=np.sqrt(np.array(cuts)-mass);xe,we=leggauss(lepton_order);xc,wc=leggauss(angle_order)
 logs={'j':[],'chi':[]};calls=0;empty=0;t0=time.monotonic()
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for x,w in zip(xe,we):
   y=lo+(x+1)*(hi-lo)/2;El=mass+y*y;ew=w*(hi-lo)*y
   for c,cw in zip(xc,wc):
    r=K.differential(state,Enu,float(El),float(c),mass,mul,antineutrino=antineutrino,model=model,energy_order=energy_order,phi_order=phi_order,tail=tail)
    calls+=1
    for name in logs:
     if r[name]['log']is not None:logs[name].append(r[name]['log']+math.log(ew*cw))
    empty+=r['chi']['log']is None
 result={name:represent(float(logsumexp(values))if values else None)for name,values in logs.items()}
 affinity=mul+b.chemical_potential-a.chemical_potential
 return dict(**result,units='s^-1',channel=f'{flavor}_{"antinu_p_to_n"if antineutrino else"nu_n_to_p"}_capture_and_inverse_ONLY',state=matter.asdict(),kernel_state=asdict(state),Enu=Enu,lepton_mass=mass,mu_lepton_effective=mul,affinity=affinity,kms_log_expected=-(Enu-affinity)/matter.temperature,kms_log_actual=result['j']['log']-result['chi']['log']if all(result[x]['log']is not None for x in logs)else None,form_factors=asdict(model),controls=dict(lepton_order=lepton_order,angle_order=angle_order,energy_order=energy_order,phi_order=phi_order,tail=tail,outer_tail=outer_tail),upper_El=upper,panels_El=(mass+cuts*cuts).tolist(),calls=calls,empty_nodes=empty,seconds=time.monotonic()-t0)
