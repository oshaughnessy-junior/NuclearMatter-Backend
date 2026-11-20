"""Independent interleaved Fierz basis and analytic elastic Born angular integrals."""
from __future__ import annotations
from pathlib import Path
from itertools import product
import importlib.util,sys,hashlib,json,math,time
import numpy as np
from scipy.integrate import quad
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'ope_nucleon_amplitudes';start=time.monotonic();checks=[];rows=[]
def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(AUTHOR.iterdir())if p.is_file()}
before=hashes();spec=importlib.util.spec_from_file_location('independent_review_ope_function',AUTHOR/'ope.py');A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
C=1.01**2/140.**2;M=939.;HC=197.3269804
basis=list(product(range(2),repeat=4)) # (s1,t1,s2,t2), unlike author's (s1,s2,t1,t2)
perm=[(2*s1+s2)*4+2*t1+t2 for s1,t1,s2,t2 in basis]
def sig(q):
 x,y,z=q;return np.array([[z,x-1j*y],[x+1j*y,-z]])
def matrix(q,k=None):
 sq=sig(q);D=np.dot(q,q)+140.**2;out=np.zeros((16,16),complex)
 sk=sig(k)if k is not None else None;E=np.dot(k,k)+140.**2 if k is not None else 1.
 for f,(s3,t3,s4,t4)in enumerate(basis):
  for i,(s1,t1,s2,t2)in enumerate(basis):
   # Pauli Fierz identity in isospin, no Pauli/Kronecker isospin matrix.
   d=2*(t3==t2)*(t4==t1)-(t3==t1)*(t4==t2)
   out[f,i]=-C*d*sq[s3,s1]*sq[s4,s2]/D
   if k is not None:
    e=2*(t4==t2)*(t3==t1)-(t4==t1)*(t3==t2)
    out[f,i]+=C*e*sk[s4,s1]*sk[s3,s2]/E
 return out

def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
points=[([31.7,-87.2,12.9],[-44.1,5.6,138.3]),([0.,0.,0.],[73.2,-19.5,8.1]),([19.7,19.7,-6.3],[19.7,19.7,-6.3]),([214.8,0.,-53.1],[0.,83.4,17.9])]
for idx,(q,k)in enumerate(points):
 q=np.array(q);k=np.array(k);ref=matrix(q,k);got=A.antisymmetrized(q,k)[np.ix_(perm,perm)]
 gate(str(idx)+' full interleaved Fierz amplitude',np.max(abs(ref-got))/C,2e-14)
 direct=matrix(q);gate(str(idx)+' direct Fierz amplitude',np.max(abs(direct-A.direct(q)[np.ix_(perm,perm)]))/C,2e-14)
 for ti in product(range(2),repeat=2):
  for tf in product(range(2),repeat=2):
   ii=[j for j,b in enumerate(basis)if(b[1],b[3])==ti];ff=[j for j,b in enumerate(basis)if(b[1],b[3])==tf];r=ref[np.ix_(ff,ii)];g=A.channel(''.join('np'[t]for t in ti),''.join('np'[t]for t in tf),q,k)
   if sum(ti)!=sum(tf):gate(str((idx,ti,tf))+' forbidden in natural C scale',np.max(abs(g))/C,1e-14)
   else:gate(str((idx,ti,tf))+' independent fixed-charge amplitude',np.max(abs(g-r))/C,2e-14)
 # Full unordered-pair permutation in the independent ordering.
 exchange=np.zeros((16,16))
 for j,(s1,t1,s2,t2)in enumerate(basis):exchange[basis.index((s2,t2,s1,t1)),j]=1
 gate(str(idx)+' outgoing coherent fermion exchange',np.max(abs(exchange@matrix(k,q)+ref))/C,2e-14)
 gate(str(idx)+' incoming coherent fermion exchange',np.max(abs(matrix(-k,-q)@exchange+ref))/C,2e-14)
 gate(str(idx)+' Hermitian amplitude',np.max(abs(ref-ref.conj().T))/C,2e-14)
 rows.append(dict(q=q.tolist(),k=k.tolist(),norm=float(np.linalg.norm(ref))))
# Coupled spin/isospin states: explicit coefficients, no matrix trace formula.
S=[np.array([1.,0.,0.,0.]),np.array([0.,1.,1.,0.])/math.sqrt(2),np.array([0.,0.,0.,1.]),np.array([0.,1.,-1.,0.])/math.sqrt(2)]
q=np.array([0.,0.,73.4]);V=matrix(q);AS=matrix(q,q)
for si,spin in enumerate(S):
 for ti,iso in enumerate(S):
  v=np.array([spin[2*s1+s2]*iso[2*t1+t2]for s1,t1,s2,t2 in basis]);sqsign=1 if si in(0,2)else-1;tv=-3 if ti==3 else 1;expected=-C*sqsign*q[2]**2/(q@q+140.**2)*tv
  gate(str((si,ti))+' spin-isospin eigenvalue',np.linalg.norm(V@v-expected*v)/C,3e-14)
  parity=(-1 if si==3 else 1)*(-1 if ti==3 else 1)
  gate(str((si,ti))+' Pauli allowed/forbidden q=k',np.linalg.norm(AS@v-(1-parity)*expected*v)/C,4e-14)
# Rotation at new axis/angle, built from 2x2 spinor and interleaved basis.
axis=np.array([.31,-.72,.27]);axis/=np.linalg.norm(axis);angle=.713;K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]);R=np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*K
U=np.eye(2)*math.cos(angle/2)-1j*sig(axis)*math.sin(angle/2);UU=np.array([[U[s3,s1]*U[s4,s2]*(t3==t1)*(t4==t2)for s1,t1,s2,t2 in basis]for s3,t3,s4,t4 in basis]);q,k=map(np.array,points[0]);gate('independent interleaved spinor rotation',np.max(abs(matrix(R@q,R@k)-UU@matrix(q,k)@UU.conj().T))/C,3e-14)
# LO flux normalization from Fermi golden rule: 2pi/(2pi)^3 *
# (Mp/2)/(2p/M)=M²/(16pi²). Independent analytic angular integrals.
def integrated(p,d,e,identical):
 a=2*p*p;b=140.**2;L=math.log1p(2*a/b)
 I=2+2*b/(b+2*a)-2*b/a*L
 J=2-b*(2*a+b)/(a*(a+b))*L
 return M*M/(16*math.pi**2)*C*C*2*math.pi*((d*d+e*e)*I+d*e*J)*HC**2*(.5 if identical else 1.)
for p in [37.4,92.1,165.3]:
 for channel,d,e,identical in [('nn',1,1,True),('pp',1,1,True),('np',-1,2,False)]:
  ref=integrated(p,d,e,identical);numeric=quad(lambda c:A.born_dsigma(channel,channel,p,c),-1,1,epsabs=1e-12,epsrel=2e-12)[0]*2*math.pi*(.5 if identical else 1.)
  gate(str((p,channel))+' absolute analytic total Born cross section',numeric/ref-1,3e-12)
  if identical:
   hemi=quad(lambda c:A.born_dsigma(channel,channel,p,c),0,1,epsabs=1e-12,epsrel=2e-12)[0]*2*math.pi;gate(str((p,channel))+' hemisphere vs final factorial',hemi/ref-1,3e-12)
  for c in [-.873,.147,.964]:
   q2=2*p*p*(1-c);k2=2*p*p*(1+c);D=q2+140.**2;F=k2+140.**2;spin=C*C*(d*d*q2*q2/D**2+e*e*k2*k2/F**2+d*e*q2*k2/(D*F));absolute=M*M/(16*math.pi**2)*spin*HC**2
   gate(str((p,channel,c))+' absolute differential Born trace',A.born_dsigma(channel,channel,p,c)/absolute-1,3e-13)
  rows.append(dict(p=p,channel=channel,analytic_total_fm2=ref,quad_total_fm2=numeric))
# Full qiqj retention: along z, central-only would give a different eigenpattern.
vup=np.array([S[0][2*s1+s2]*S[0][2*t1+t2]for s1,t1,s2,t2 in basis]);vzero=np.array([S[1][2*s1+s2]*S[0][2*t1+t2]for s1,t1,s2,t2 in basis]);gate('retained anisotropic triplet tensor eigenpattern',float(np.real((vzero@V@vzero)/(vup@V@vup)))+1,2e-14)
gate('on-shell pseudoscalar convention',A.Model().pseudoscalar_g/(2*939*1.01/140)-1,2e-15)
gate('frozen author all artifacts unchanged',int(hashes()!=before),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,author_before=before,author_after=hashes(),seconds=time.monotonic()-start,scope='declared static NR unregulated strong OPE Born model only; no empirical NN or modified-Urca rate')
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in ['passed','failed','seconds']},indent=2))
