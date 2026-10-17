from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
import scatter as S
P=Path(__file__).resolve().parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for flavor in ['e','mu']:
 C=S.P.Couplings(flavor)
 for pos in [False,True]:
  for anti in [False,True]:
   for E in [.3,10.]:
    ss=S.ME*S.ME+2*S.ME*E;rt=math.sqrt(ss);p=(ss-S.ME**2)/(2*rt);et=(ss+S.ME**2)/(2*rt);x,w=leggauss(12);sigma=0.
    for c,ww in zip(x,w):
     k,pv,kp,pp=[np.array([v],float)for v in [[p,0,0,p],[et,0,0,-p],[p,p*math.sqrt(1-c*c),0,p*c],[et,-p*math.sqrt(1-c*c),0,-p*c]]]
     scalar=float(S.amplitude(k,pv,kp,pp,C,positron=pos,anti=anti)[0]);spin=float(S.amplitude(k,pv,kp,pp,C,positron=pos,anti=anti,method='spinor')[0]);gate('literal crossing spinor/scalar '+str((flavor,pos,anti,E)),scalar/spin-1,3e-11);sigma+=ww*scalar/(32*math.pi*ss)
    Tmax=2*E*E/(S.ME+2*E);ref=quad(lambda t:S.vacuum_dsigma_drecoil(E,t,C,pos,anti),0,Tmax,epsabs=1e-40,epsrel=2e-12)[0];gate('absolute recoil/CM cross section',sigma/ref-1,3e-11);rows.append(dict(flavor=flavor,positron=pos,anti=anti,E=E,sigma_MeV_minus2=sigma,sigma_cm2=sigma*S.HC*S.HC*1e-26))
(P/'vacuum_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=all(c['passed']for c in checks)),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']][:3])
