"""Independent moving-target vacuum-flux anchor, no final-state blocking.

Uses target kinetic-energy Laguerre nodes, incoming target-angle quadrature,
and source-invariant vacuum CM angle integration, not medium tensor geometry.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import roots_genlaguerre,expit
from numpy.polynomial.legendre import leggauss
from capture import K

def anchor(species,Enu,ml,mb,*,antineutrino=False,radial_order=16,target_angle_order=12,cm_angle_order=16,model=K.C.FormFactors()):
 T,M,mu=species.temperature,species.mass,species.chemical_potential
 x,w=roots_genlaguerre(radial_order,.5);c,wc=leggauss(target_angle_order)
 channel=K.V.Channel(M,mb,ml,antineutrino,model.gv,model.ga)
 weights=[];values=[]
 for xx,ww in zip(x,w):
  Ea=M+T*xx;p=math.sqrt(T*xx*(2*M+T*xx))
  # sqrt(x) is already in the Laguerre weight; exp(x) undone stably.
  logf=float(-np.logaddexp(0.,(Ea-mu)/T))
  weight=ww*T*math.sqrt(T*(2*M+T*xx))*Ea*math.exp(xx+logf)
  fluxsum=0.
  for cc,www in zip(c,wc):
   flux=1-p/Ea*cc;Er=Enu*Ea/M*flux
   fluxsum+=www/2*flux*K.C.total(channel,float(Er),model,order=cm_angle_order,method='source')
  weights.append(weight);values.append(weight*fluxsum)
 density=sum(weights)/(math.pi**2*K.V.HBARC**3)
 mean=sum(values)/sum(weights)
 return dict(density_integral=density,density_EOS=species.density,density_ratio=density/species.density,mean_flux_sigma_cm2=mean,chi_s=species.density*1e39*2.99792458e10*mean,controls=dict(radial_order=radial_order,target_angle_order=target_angle_order,cm_angle_order=cm_angle_order),blocking='No final baryon/lepton blocking; valid comparison requires dilute final occupations',method='FD target average of source-invariant full-current vacuum cross section with Moller flux')
