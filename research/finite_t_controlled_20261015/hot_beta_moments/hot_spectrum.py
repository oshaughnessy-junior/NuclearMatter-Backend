"""NEW selected T2/.42 reference: low-T frozen API is NOT extended.

Copied integration geometry from frozen kernel; new forward/refinement gates.
"""
import math,time
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss
from dataclasses import asdict
from hot_moments import K
MN=K.MN;MP=K.MP;ME=K.ME;t=K.t
represented=K.represented;differential=K.differential
def spectral(bath,Enu,*,angle_order=8,lepton_order=12,widths=9.,energy_order=12,phi_order=4,tail=50.,model=t.C.FormFactors(),budget_seconds=110.):
 """Integrate selected sub-rest-endpoint energy with Doppler-resolved Ee panels.

 Ee truncation is empirical/refinement-qualified, not a universal tail bound.
 Differential API has full invariant support including moving endpoint tails.
 """
 if (bath.T,bath.mu_n,bath.mu_p,bath.mu_e)!=(2.,919.8287578100662,916.8449303084382,.17446333200826397)or Enu!=.42:raise ValueError('New exact selected T2/nB1e-7/Ye.3/Enu.42 numerical reference only')
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
