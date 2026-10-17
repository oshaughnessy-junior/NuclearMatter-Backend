"""New bounded whole-lepton integration of frozen crossed physical tensor.

No changes to old low-T API or old .42-only hot reference.
"""
import math,time
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss

def reference(S,E,angle=10,lepton=12,energy=20,phi=6,widths=18.,tail=80.,budget=105.):
 b,_=S.matched();x,w=leggauss(angle);y,v=leggauss(lepton);logs={'j':[],'chi':[]};panels=[];start=time.monotonic();calls=0
 for c,wc in zip(x,w):
  if E<S.ENDPOINT:
   def shell(El):
    pe=math.sqrt((El-S.ME)*(El+S.ME));return E+El+math.sqrt(S.MP**2+pe*pe+E*E+2*pe*E*c)-S.MN
   center=brentq(shell,S.ME,S.MN-S.MP,xtol=2e-13);pe=math.sqrt((center-S.ME)*(center+S.ME));q=math.sqrt(E*E+pe*pe+2*E*pe*c);width=q*math.sqrt(b.T/S.MN)
   cuts=sorted({S.ME,center}|{max(S.ME,center+h*width)for h in[-widths,-8,-3,-1,1,3,8,widths]})
  else:
   center=S.ME;width=E*math.sqrt(b.T/S.MN);cuts=sorted({S.ME+h*width for h in[0,.05,.2,.5,1,2,4,8,widths]})
  # Avoid largeirrelevantinterval torestenergy if numericaltail loweredge>me.
  if E<S.ENDPOINT and cuts[1]>S.ME and center-widths*width>S.ME:cuts=cuts[1:]
  panels.append(dict(c=float(c),center=center,width=width,edges=cuts))
  for lo,hi in zip(cuts[:-1],cuts[1:]):
   a,z=math.sqrt(lo-S.ME),math.sqrt(hi-S.ME)
   for t,wt in zip((a+z)/2+(z-a)*y/2,v*(z-a)):
    El=S.ME+t*t;weight=wc*wt*t
    if time.monotonic()-start>budget:raise TimeoutError('Newcrossedtensorfull-lepton budget')
    d=S.K.differential(b,E,float(El),float(c),energy_order=energy,phi_order=phi,tail=tail);calls+=1
    for key in logs:
     if d[key]['log']is not None:logs[key].append(math.log(weight)+d[key]['log'])
 return dict(Enu=E,**{k:S.represented(float(logsumexp(v))if v else None)for k,v in logs.items()},panels=panels,controls=dict(angle=angle,lepton=lepton,energy=energy,phi=phi,widths=widths,tail=tail),calls=calls,seconds=time.monotonic()-start)
