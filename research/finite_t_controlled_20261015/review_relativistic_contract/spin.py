"""Explicit massless active/inactive spin-density projector normalization."""
from pathlib import Path
import numpy as np,json,math
P=Path(__file__).resolve().parent
I=np.eye(2);Z=np.zeros((2,2));sig=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])];g=[np.block([[I,Z],[Z,-I]])]+[np.block([[Z,s],[-s,Z]])for s in sig];g5=1j*g[0]@g[1]@g[2]@g[3];PL=(np.eye(4)-g5)/2;PR=(np.eye(4)+g5)/2
slash=lambda p:g[0]*p[0]-sum((g[i+1]*p[i+1]for i in range(3)),np.zeros((4,4),complex))
checks=[]
for E,ml,lv in [(77.,.511,np.array([13.,-22.,34.])),(214.,105.658,np.array([-41.,79.,17.]))]:
 kv=np.array([.3,-.4,math.sqrt(.75)])*E;k=np.r_[E,kv];l=np.r_[math.sqrt(ml*ml+lv@lv),lv]
 for mu in range(4):
  for nu in range(4):
   vertex=g[mu]@(np.eye(4)-g5);conjugate=g[nu]@(np.eye(4)-g5);out=slash(l)+ml*np.eye(4);ks=slash(k)
   full=np.trace(out@vertex@ks@conjugate);active=np.trace(out@vertex@(PL@ks)@conjugate);inactive=np.trace(out@vertex@(PR@ks)@conjugate);scale=max(abs(full),1.)
   for label,error in [('full equals active',abs(full-active)/scale),('inactive vanishes',abs(inactive)/scale)]:checks.append(dict(name=str((E,mu,nu))+label,error=float(error),tolerance=3e-13,passed=bool(error<3e-13)))
(P/'spin_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
