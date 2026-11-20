"""New interleaved explicit-state leg sums, kinetic resolvents and Ward algebra."""
from __future__ import annotations
from pathlib import Path
from itertools import product
import importlib.util,sys,hashlib,json,math,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;AUTHOR=ROOT/'ope_weak_diagrams';checks=[];rows=[];start=time.monotonic()
def snapshot():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for d in['ope_weak_diagrams','ope_nucleon_amplitudes']for p in(ROOT/d).rglob('*')if p.is_file()and'__pycache__'not in str(p)}
before=snapshot();spec=importlib.util.spec_from_file_location('review_only_ope_weak_external',AUTHOR/'external_legs.py');W=importlib.util.module_from_spec(spec);sys.modules[spec.name]=W;spec.loader.exec_module(W)
BASIS=list(product(range(2),repeat=4));PERM=[(2*s1+s2)*4+2*t1+t2 for s1,t1,s2,t2 in BASIS];M=939.;C=1.01**2/140.**2
P=np.zeros((16,16))
for j,(s,t,u,v)in enumerate(BASIS):P[BASIS.index((u,v,s,t)),j]=1

def gate(n,e,t=3e-12):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def compare(n,a,b,scale=None,tol=3e-12):gate(n,np.max(abs(a-b))/(max(np.max(abs(a)),np.max(abs(b)),1e-100)if scale is None else scale),tol)
def sigma(q):
 x,y,z=q;return np.array([[z,x-1j*y],[x+1j*y,-z]])
def strong(q):
 s=sigma(q);out=np.zeros((16,16),complex)
 for f,(a,b,c,d)in enumerate(BASIS):
  for i,(e,h,g,j)in enumerate(BASIS):
   td=2*(b==j)*(d==h)-(b==h)*(d==j);out[f,i]=-C*s[a,e]*s[c,g]*td/(q@q+140.**2)
 return out

def weak_actions(state,line,component,momentum,Q,after,raising):
 # Act literally on one external 4-state nucleon; return its possible
 # intermediate states. No weak 16x16 matrix multiplication construction.
 s=list(state);spin_index=2*line;iso_index=2*line+1
 if s[iso_index]!=(1 if raising else 0):return []
 s[iso_index]=0 if raising else 1
 if component=='V0':return [(tuple(s),1.)]
 if component[0]=='V':
  a='xyz'.index(component[1]);factor=(2*momentum[a]+(1 if after else -1)*Q[a])/(2*M);return [(tuple(s),factor)]
 sig=sigma(np.eye(3)['xyz'.index(component[1])]);oldspin=state[spin_index];out=[]
 for spin in range(2):
  z=s.copy();z[spin_index]=spin;v=-1.26*sig[spin,oldspin]
  if v!=0:out.append((tuple(z),v))
 return out

def direct(p,omega,Q,component,raising=False):
 E=lambda v:sum(float(z)**2 for z in v)/(2*M)
 V=[strong(p[3]-p[1]),strong(p[2]-p[0])];out=np.zeros((16,16),complex);den=[]
 for line in[0,1]:
  di=E(p[line])-omega-E(p[line]-Q);do=E(p[line+2])+omega-E(p[line+2]+Q);den.extend([di,do])
  for i,ins in enumerate(BASIS):
   # Incoming emission changes the ket BEFORE strong scattering.
   for mid,w in weak_actions(ins,line,component,p[line],Q,False,raising):out[:,i]+=V[line][:,BASIS.index(mid)]*w/di
   # Outgoing emission acts on each literal strong outgoing intermediate ket.
   for midindex,mid in enumerate(BASIS):
    for final,w in weak_actions(mid,line,component,p[line+2],Q,True,raising):out[BASIS.index(final),i]+=w*V[line][midindex,i]/do
 return out,den

def amplitude(p,omega,Q,comp,raising=False):
 a,den=direct(p,omega,Q,comp,raising);ex,_=direct(p[[0,1,3,2]],omega,Q,comp,raising);return a-P@ex,den

def point(pmag,omega,Q,u):
 # Solve conservation independently as relative final momentum and CM recoil.
 Q=np.array(Q,float);u=np.array(u,float);r=math.sqrt(pmag**2-M*omega-Q@Q/4)*u/np.linalg.norm(u)
 return np.array([[0.,0.,pmag],[0.,0.,-pmag],r-Q/2,-r-Q/2]),Q

def charge(line,raising=False):
 out=np.zeros((16,16))
 for i,s in enumerate(BASIS):
  for final,v in weak_actions(s,line,'V0',np.zeros(3),np.zeros(3),False,raising):out[BASIS.index(final),i]=v
 return out
points=[(117.3,6.7,[-1.7,2.9,.8],[.24,-.61,.75]),(83.1,2.8,[1.4,.7,-2.1],[-.46,.32,.83])]
for idx,args in enumerate(points):
 p,Q=point(*args);om=args[1];gate(str(idx)+' independent kinetic energy conservation',(np.sum(p[:2]**2)-np.sum(p[2:]**2))/(2*M)-om,2e-13);gate(str(idx)+' momentum conservation',np.max(abs(p[0]+p[1]-p[2]-p[3]-Q)),5e-14)
 measured={};all_terms={}
 for raising in[False,True]:
  for comp in['V0','Vx','Vy','Vz','Ax','Ay','Az']:
   a,den=amplitude(p,om,Q,comp,raising);got=W.amplitude(p,om,Q,comp,raising=raising)[np.ix_(PERM,PERM)];compare(str((idx,raising,comp))+' literal leg-action forward',a,got);measured[(raising,comp)]=a
   # Charge selection total proton count must change by exactly one.
   for ti in product(range(2),repeat=2):
    for tf in product(range(2),repeat=2):
     if sum(tf)-sum(ti)!=( -1 if raising else 1):
      ii=[i for i,s in enumerate(BASIS)if(s[1],s[3])==ti];ff=[i for i,s in enumerate(BASIS)if(s[1],s[3])==tf];gate(str((idx,raising,comp,ti,tf))+' forbidden charged block',np.max(abs(got[np.ix_(ff,ii)]))/(C/min(abs(x)for x in den)),3e-13)
 for raising in[False,True]:
  def rhs(pp):
   Vs=[strong(pp[3]-pp[1]),strong(pp[2]-pp[0])];return sum(charge(i,raising)@Vs[i]-Vs[i]@charge(i,raising)for i in[0,1])
  defect=rhs(p)-P@rhs(p[[0,1,3,2]]);lhs=om*measured[(raising,'V0')]-sum(Q[j]*measured[(raising,'V'+a)]for j,a in enumerate('xyz'))
  compare(str((idx,raising))+' independent finiteQ Ward defect',lhs,defect);gate(str((idx,raising))+' finiteQ defect explicitly NONZERO',float(np.max(abs(defect))<1e-8*C),0)
  helper=W.vector_defect_direct(p,om,Q,raising=raising);sw=W.vector_defect_direct(p[[0,1,3,2]],om,Q,raising=raising);helper=(helper-W.ope.EXCHANGE@sw)[np.ix_(PERM,PERM)];compare(str((idx,raising))+' exposed defect forward',helper,defect)
  all_terms[str(raising)]=dict(divergence_max=float(np.max(abs(defect))),divergence_frobenius=float(np.linalg.norm(defect)),V0_frobenius=float(np.linalg.norm(measured[(raising,'V0')])) )
 rows.append(dict(p=p.tolist(),omega=om,Q=Q.tolist(),kinetic_resolvents=amplitude(p,om,Q,'V0')[1],weak=all_terms))
# Independent soft commutator, no author's strong or weak operator calls.
p,Q=point(96.4,4.1,[0,0,0],[.13,.73,-.42]);om=4.1;Vs=strong(p[2]-p[0])-P@strong(p[3]-p[0]);Jv=charge(0)+charge(1)
compare('soft total charged-isospin scalar commutator',Jv@Vs,Vs@Jv,scale=C)
compare('soft vector cancellation',W.amplitude(p,om,Q,'V0')[np.ix_(PERM,PERM)],np.zeros((16,16)),scale=C/om)
for comp in['Ax','Ay','Az']:
 J=np.zeros((16,16),complex)
 for i,s in enumerate(BASIS):
  for line in[0,1]:
   for final,v in weak_actions(s,line,comp,np.zeros(3),Q,False,False):J[BASIS.index(final),i]+=v
 expected=(J@Vs-Vs@J)/om;actual=W.amplitude(p,om,Q,comp)[np.ix_(PERM,PERM)];compare('soft '+comp+' independently constructed commutator',actual,expected);gate('soft '+comp+' retained NONZERO',float(np.linalg.norm(expected)<1e-8*C/om),0)
gate('frozen source and preserved failures unchanged',int(snapshot()!=before),0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],points=rows,author_before=before,author_after=snapshot(),seconds=time.monotonic()-start,scope='offpole external one-body weak legs only; incomplete finiteQ current, no physical MU rate')
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','seconds']},indent=2))
