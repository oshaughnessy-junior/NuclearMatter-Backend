"""Independent support interval witnesses, including timelike and null transfer."""
from pathlib import Path
import json,math,numpy as np
P=Path(__file__).resolve().parent;checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for ma,mb,q,w in [(939.,938.,53.,-17.),(700.,810.,20.,105.),(810.,700.,20.,-105.),(939.,939.,50.,60.),(939.,939.,50.,-60.),(700.,810.,30.,30.),(810.,700.,30.,-30.)]:
 C=ma*ma-mb*mb+w*w-q*q;A=q*q-w*w;B=-w*C;D=-(C*C/4+q*q*ma*ma)
 roots=sorted([float(x.real)for x in np.roots([A,B,D]if A else [B,D])if abs(x.imag)<1e-9]);positivefloor=max(ma,mb-w)
 accepted=[];mismatch=0
 for E in np.linspace(positivefloor+1e-5,positivefloor+3500,513):
  p=math.sqrt(E*E-ma*ma);x=(C+2*w*E)/(2*p*q);physical=abs(x)<=1 and E+w>=mb
  polynomial=A*E*E+B*E+D>=0
  mismatch+=physical!=polynomial
  if physical:accepted.append(E)
 gate('polynomial/originalangle full support '+str((ma,mb,q,w)),mismatch,.1)
 for E in roots:
  if E>positivefloor:
   p=math.sqrt(E*E-ma*ma);x=(C+2*w*E)/(2*p*q);reconstructed=math.sqrt(mb*mb+p*p+q*q+2*p*q*x)
   gate('root support angle endpoint '+str((ma,mb,q,w,E)),abs(abs(x)-1),2e-11)
   gate('root original energy endpoint '+str((ma,mb,q,w,E)),abs(reconstructed-(E+w)),3e-10)
 rows.append(dict(ma=ma,mb=mb,q=q,w=w,transfer='spacelike'if A>0 else('timelike'if A<0 else'lightlike'),quadratic_roots=roots,positive_energy_floor=positivefloor,sampled_support_count=len(accepted),sampled_min=min(accepted)if accepted else None,sampled_max=max(accepted)if accepted else None))
(P/'support_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
