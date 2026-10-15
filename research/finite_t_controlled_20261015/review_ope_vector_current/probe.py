"""Interleaved-basis Hamiltonian derivative and two-component source limits."""
from pathlib import Path
import sys,importlib.util,json,math,hashlib,time
from itertools import product
import numpy as np
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'ope_vector_current';sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('review_static_vector',AUTHOR/'current.py');C=importlib.util.module_from_spec(sp);sys.modules[sp.name]=C;sp.loader.exec_module(C)
before={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in AUTHOR.iterdir()if x.is_file()};checks=[];start=time.monotonic()
BASIS=list(product(range(2),repeat=4));PERM=[(2*s1+s2)*4+2*t1+t2 for s1,t1,s2,t2 in BASIS]
P=np.zeros((16,16));
for i,(s1,t1,s2,t2)in enumerate(BASIS):P[BASIS.index((s2,t2,s1,t1)),i]=1
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.]);pauli=[sx,sy,sz];I=np.eye(2);lower=np.array([[0,0],[1,0]],complex)
M=939.;mp=140.;g=1.01/mp;coef=g*g
def eps(a,b,c):return 0 if len({a,b,c})<3 else (1 if (a,b,c)in[(0,1,2),(1,2,0),(2,0,1)]else -1)
def lift(line,s,t):
 out=np.zeros((16,16),complex)
 for f,fs in enumerate(BASIS):
  for i,ins in enumerate(BASIS):
   if fs[2*(1-line):2*(1-line)+2]!=ins[2*(1-line):2*(1-line)+2]:continue
   out[f,i]=s[fs[2*line],ins[2*line]]*t[fs[2*line+1],ins[2*line+1]]
 return out
def sig(q):return sum(q[a]*pauli[a]for a in range(3))
def rebase(a):return np.array(a)[...,PERM,:][...,PERM]
def ck(name,error,tol=3e-12):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
def compare(name,a,b,scale=None,tol=3e-12):
 scale=max(np.max(abs(a)),np.max(abs(b)),1e-30)if scale is None else scale
 ck(name,np.max(abs(a-b))/scale,tol)

def variation(k1,k2,raising=False):
 # Build pion fields and source tensors separately. Differentiate the TWO
 # original Hamiltonian terms, not the compact commutator or Ward residual.
 ks=[np.array(k1),np.array(k2)];phi=[[ -1j*g*lift(l,sig(ks[l]),pauli[a])/(ks[l]@ks[l]+mp*mp)for a in range(3)]for l in range(2)]
 sea=np.zeros((3,16,16),complex);flight=sea.copy();charged=[1,1j if raising else -1j,0]
 for i,a,b,c in product(range(3),repeat=4):
  e=eps(a,b,c)*charged[b]
  if not e:continue
  # -dH/dv: vertex term g S.Dpi, gradient term (Dpi)^2/2.
  sea[i]-=g*e*(lift(0,pauli[i],pauli[a])@phi[1][c]+lift(1,pauli[i],pauli[a])@phi[0][c])
  flight[i]-=e*((-1j*ks[0][i]*phi[0][a])@phi[1][c]+(-1j*ks[1][i]*phi[1][a])@phi[0][c])
 return dict(seagull=sea,pion_flight=flight,total=sea+flight)

def own_A(p,raising=False):
 a=variation(p[2]-p[0],p[3]-p[1],raising);b=variation(p[3]-p[0],p[2]-p[1],raising)
 return {name:a[name]-np.array([P@x for x in b[name]])for name in a}
def strong(k):return -coef*sum(lift(0,sig(k),t)@lift(1,sig(k),t)for t in pauli)/(k@k+mp*mp)

rows=[]
for j,(pmag,om,Q,u)in enumerate([(109.3,4.9,[3.2,-2.1,1.4],[.41,-.63,.17]),(142.8,8.6,[-1.9,.8,3.7],[.23,.37,-.61])]):
 Q=np.array(Q);p=C.W.emission_point(pmag,om,Q,u)
 for raising in [False,True]:
  orig=C.antisymmetrized(p,raising=raising);ind=own_A(p,raising)
  direct=variation(p[2]-p[0],p[3]-p[1],raising)
  raw=C.direct(p[2]-p[0],p[3]-p[1],raising=raising)
  for name in ind:
   compare(str((j,raising,name))+' direct Hamiltonian variation',rebase(raw[name]),direct[name])
   compare(str((j,raising,name))+' final-exchanged Hamiltonian variation',rebase(orig[name]),ind[name])
   compare(str((j,raising,name))+' incoming exchange',np.array([x@P for x in rebase(C.antisymmetrized(p[[1,0,2,3]],raising=raising)[name])]),-ind[name])
   compare(str((j,raising,name))+' outgoing exchange',np.array([P@x for x in rebase(C.antisymmetrized(p[[0,1,3,2]],raising=raising)[name])]),-ind[name])
   reverse=rebase(C.antisymmetrized(p[[2,3,0,1]],raising=not raising)[name])
   compare(str((j,raising,name))+' charge adjoint',reverse.conj().transpose(0,2,1),ind[name])
  R=[lift(l,I,lower.conj().T if raising else lower)for l in (0,1)]
  v2=strong(p[3]-p[1]);v1=strong(p[2]-p[0]);defect=R[0]@v2-v2@R[0]+R[1]@v1-v1@R[1]
  ps=p[[0,1,3,2]];v2s=strong(ps[3]-ps[1]);v1s=strong(ps[2]-ps[0]);defect-=P@(R[0]@v2s-v2s@R[0]+R[1]@v1s-v1s@R[1])
  contraction=np.einsum('i,ijk->jk',Q,ind['total'])
  compare(str((j,raising))+' independently assembled potential continuity',contraction,defect)
  ext=rebase(C.combined(p,om,Q,raising=raising)['external']);allcur=rebase(C.combined(p,om,Q,raising=raising)['total'])
  scale=max(np.max(abs(defect)),1e-30)
  ck(str((j,raising))+' whole static vector continuity',np.max(abs(om*allcur[0]-np.einsum('i,ijk->jk',Q,allcur[1:])))/scale,3e-12)
  for name in ['seagull','pion_flight']:
   remaining=defect-np.einsum('i,ijk->jk',Q,ind[name]);ck(str((j,raising,name))+' one piece insufficient',float(np.max(abs(remaining))<1e-5*scale),0)
  transverse=ind['total']-Q[:,None,None]*contraction[None,:,:]/(Q@Q)
  ck(str((j,raising))+' genuine nonzero transverse part',float(np.max(abs(transverse))<1e-8*np.max(abs(ind['total']))),0)
  rows.append(dict(p=p.tolist(),Q=Q.tolist(),omega=om,raising=raising,original_defect_max=float(np.max(abs(defect))),total_divergence_max=float(np.max(abs(om*allcur[0]-np.einsum('i,ijk->jk',Q,allcur[1:])))),transverse_max=float(np.max(abs(transverse)))))

# Independent two-component evaluation of the primary relativistic bilinears.
# ubar u=1. No gamma matrices or author relativistic_limit.py is imported.
def bilinears(pi,pf):
 Ei=math.sqrt(M*M+pi@pi);Ef=math.sqrt(M*M+pf@pf)
 Bi=math.sqrt((Ei+M)/(2*M))*I;Bf=math.sqrt((Ef+M)/(2*M))*I
 Si=sig(pi)/math.sqrt(2*M*(Ei+M));Sf=sig(pf)/math.sqrt(2*M*(Ef+M))
 J=[-Bf@s@Bi-Sf@s@Si for s in pauli]
 J0=-Bf@Si-Sf@Bi;k=pf-pi;w=Ef-Ei
 vertex=(w*J0-sum(k[i]*J[i]for i in range(3)))/(w*w-k@k-mp*mp)
 return J,vertex
def source_rel(p,raising):
 J1,V1=bilinears(p[0],p[2]);J2,V2=bilinears(p[1],p[3]);k1=p[2]-p[0];k2=p[3]-p[1];sea=np.zeros((3,16,16),complex);flight=sea.copy();charged=[1,1j if raising else -1j,0]
 for b,a,c in product(range(3),repeat=3):
  fac=1j*eps(b,a,c)*charged[b]*coef
  if not fac:continue
  for i in range(3):
   sea[i]+=fac*(lift(0,V1,pauli[a])@lift(1,J2[i],pauli[c])-lift(0,J1[i],pauli[a])@lift(1,V2,pauli[c]))
   flight[i]+=fac*(k1[i]-k2[i])*lift(0,V1,pauli[a])@lift(1,V2,pauli[c])
 return dict(seagull=sea,pion_flight=flight)
p0=np.array([[37.,-19.,81.],[-29.,11.,-63.],[15.,31.,-47.],[-21.,-17.,35.]])
limit=[]
for raising in [False,True]:
 prev={}
 for scale in [.8,.4,.2,.1]:
  p=p0*scale;actual=source_rel(p,raising);static=variation(p[2]-p[0],p[3]-p[1],raising);errs={}
  for name in actual:
   error=float(np.linalg.norm(actual[name]-static[name])/np.linalg.norm(static[name]));errs[name]=error
   if prev:ck(str((raising,scale,name))+' fresh source NR convergence',error/prev[name],.6)
   if scale==.1:ck(str((raising,name))+' final source static limit',error,2e-4)
  prev=errs;limit.append(dict(raising=raising,scale=scale,errors=errs))

for k1,k2 in [(np.array([12.,-21.,17.]),np.array([-12.,21.,-17.])),(np.zeros(3),np.array([15.,7.,-11.])),(np.zeros(3),np.zeros(3))]:
 for raising in [False,True]:
  a=C.direct(k1,k2,raising=raising);b=variation(k1,k2,raising)
  for name in a:compare('soft/onezero/twozero '+str((tuple(k1),tuple(k2),raising,name)),rebase(a[name]),b[name],scale=coef/mp)
after={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in AUTHOR.iterdir()if x.is_file()};ck('all frozen author artifacts unchanged',float(before!=after),0)
result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],points=rows,source_limit=limit,seconds=time.monotonic()-start,author_before=before,author_after=after,scope='minimally gauged instantaneous static model vector current only; no complete weak rate or uncertainty')
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','failed','seconds']}))
