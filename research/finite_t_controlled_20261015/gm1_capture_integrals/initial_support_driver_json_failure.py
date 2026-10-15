"""Exact equal-mass particle-hole angular support and phase Jacobians."""
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
from pathlib import Path
import capture as C
checks=[]
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=abs(e)<=t))
for mass in(350.,568.5572415829281,939.):
 for w in(-150.,-20.,0.,20.,150.):
  for factor in(.9,1.,1.1):
   if w==0 and factor!=1:continue
   q=abs(w)*factor if w else 20.
   support=C.A.free.support(mass,mass,w,q)
   ck('equal_mass_support',int(bool(support))-int(q>abs(w)),0.)
x,wx=leggauss(16)
for Enu in(40.,100.,200.):
 for El in(106.,140.,220.):
  pl=math.sqrt(El*El-C.A.leptons.MUON_MASS**2)
  for dU in(-60.,0.,60.):
   w=Enu-El+dU;qlo=max(abs(Enu-pl),abs(w));qhi=Enu+pl
   if qlo>=qhi:continue
   cmax=min(1.,(Enu*Enu+pl*pl-w*w)/(2*Enu*pl))
   q=qlo+(x+1)*(qhi-qlo)/2;c=(Enu*Enu+pl*pl-q*q)/(2*Enu*pl);weight=wx*(qhi-qlo)/2*q/(Enu*pl)
   ck('q_coordinate_measure',sum(weight)-(cmax+1),2e-13)
   ck('q_coordinate_first_moment',sum(weight*c)-(cmax*cmax-1)/2,2e-13)
   ck('q_coordinate_second_moment',sum(weight*c*c)-(cmax**3+1)/3,2e-13)
out=dict(checks=checks,passed=sum(v['passed']for v in checks),failed=sum(not v['passed']for v in checks))
(Path(__file__).resolve().parent/'support_checks.json').write_text(json.dumps(out,indent=2));print(out['passed'],out['failed'])
if out['failed']:raise SystemExit(1)
