"""Independently ordered outgoing-energy/cosine integral of exact-shell kernel."""
from pathlib import Path
import json,time,math
import numpy as np
from numpy.polynomial.legendre import leggauss
import scatter as S
P=Path(__file__).resolve().parent;C=S.P.Couplings('e');rows=[];checks=[]
def integrate(order,angular,upper,inner):
 start=time.monotonic();x,g=leggauss(order);z,h=leggauss(angular);y=(z+1)*math.sqrt(2)/2;yw=h*math.sqrt(2)/2*2*y;v=np.zeros(4);calls=0
 for yy,ww in zip(y,yw):
  c=1-yy*yy;width=10*yy;edges=sorted(set([0.,upper,1.,3.,6.,10.,15.,25.,45.,75.]+[t for scale in [.25,.5,1,2,4]for t in [10-width*scale,10+width*scale]if 0<t<upper]));edges=[t for t in edges if t<=upper]
  for lo,hi in zip(edges[:-1],edges[1:]):
   for u,w in zip((lo+hi)/2+(hi-lo)*x/2,g*(hi-lo)/2):
    if time.monotonic()-start>110:raise TimeoutError('Independent shellkernel momentintegral budget')
    r=S.differential(2.,.17446333200826397,10.,float(u),float(c),C,order=inner,phi=4,tail=70.);val=r['value']*w*ww;v+=val*np.array([1,1-c,u-10,u]);calls+=1
 return dict(values=dict(zip(['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment'],v.tolist())),controls=dict(order=order,angle=angular,upper=upper,inner=inner,angular_coordinate='sqrt(1-c)'),seconds=time.monotonic()-start,calls=calls)
def save():
 (P/'kernel_integral_results.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks)),indent=2)+'\n')
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for controls in [(10,12,110.,20),(16,20,150.,28)]:
 r=integrate(*controls);rows.append(r);save()
for key in rows[-1]['values']:gate('shellintegralrefine '+key,rows[0]['values'][key]/rows[1]['values'][key]-1,3e-6)
save();print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
