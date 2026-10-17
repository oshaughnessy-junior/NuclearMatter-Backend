"""Independent Dirac-bilinear leading-static limit of Simo vector Eqs17,26.

ubar u=1 states, F1V=1, pion vertices unregulated. This is a LIMIT check,
not a qualified finite-momentum implementation of the full Simo model.
"""
from __future__ import annotations
from pathlib import Path
import json,math,time
import numpy as np
import current as C
HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
I=np.eye(2);Z=np.zeros((2,2));P=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1,-1])];G=np.array([np.diag([1,1,-1,-1])]+[np.block([[Z,s],[-s,Z]])for s in P]);G5=1j*G[0]@G[1]@G[2]@G[3];M=939.;mpi=140.;coefficient=1.01**2/mpi**2

def spinor(p):
 E=math.hypot(M,float(np.linalg.norm(p)));return math.sqrt((E+M)/(2*M))*np.vstack([I,sum(p[i]*P[i]for i in range(3))/(E+M)])
def eps(a,b,c):return 0 if len({a,b,c})<3 else(1 if(a,b,c)in[(0,1,2),(1,2,0),(2,0,1)]else -1)
def source_current(p,raising=False):
 V=[];J=[];ks=[]
 for line in[0,1]:
  pi,pf=p[line],p[line+2];Ei,Ef=math.hypot(M,np.linalg.norm(pi)),math.hypot(M,np.linalg.norm(pf));k=np.array([Ef-Ei,*(pf-pi)]);slash=k[0]*G[0]-sum(k[i+1]*G[i+1]for i in range(3));left=spinor(pf).conj().T@G[0];right=spinor(pi);den=k[0]**2-k[1:]@k[1:]-mpi**2
  V.append(left@G5@slash@right/den);J.append([left@G5@G[i+1]@right for i in range(3)]);ks.append(k)
 cross=[sum(eps(a,b,c)*np.kron(P[b],P[c])for b in range(3)for c in range(3))for a in range(3)];iso=1j*cross[0]+(1j if raising else -1j)*(1j*cross[1]);sea=[];flight=[]
 for i in range(3):
  sea.append(coefficient*np.kron(np.kron(V[0],J[1][i])-np.kron(J[0][i],V[1]),iso))
  flight.append(coefficient*(ks[0][i+1]-ks[1][i+1])*np.kron(np.kron(V[0],V[1]),iso))
 return dict(seagull=np.array(sea),pion_flight=np.array(flight)),V,J
p0=C.W.emission_point(128.7,7.3,[2.7,.4,-1.9],[.63,-.27,.42])
for raising in[False,True]:
 previous={}
 for lam in[1.,.5,.25,.125]:
  p=p0*lam;rel,V,J=source_current(p,raising);nr=C.direct(p[2]-p[0],p[3]-p[1],raising=raising);errors={}
  for name in['seagull','pion_flight']:
   err=float(np.linalg.norm(rel[name]-nr[name])/np.linalg.norm(nr[name]));errors[name]=err
   if lam<1.:checks.append(dict(name=str((raising,lam,name))+' primary Dirac NR error decreases',error=err/previous[name],tolerance=.65,passed=err/previous[name]<=.65))
   if lam==.125:checks.append(dict(name=str((raising,name))+' final primary NR limit',error=err,tolerance=5e-4,passed=err<=5e-4))
  previous=errors;rows.append(dict(raising=raising,scale=lam,errors=errors))
C.verify_sources();out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,pins=C.PINS,seconds=time.monotonic()-start,scope='leading spatial static NR limit, not exact finitep Simo model/current normalization certification')
(HERE/'relativistic_limit_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','rows','seconds']},indent=2))
