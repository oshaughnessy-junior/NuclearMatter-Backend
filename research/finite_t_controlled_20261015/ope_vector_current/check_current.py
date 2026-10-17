"""Independent gauged-vertex epsilon contraction and selected continuity tests."""
from __future__ import annotations
import math,json,time,hashlib
from pathlib import Path
import numpy as np
import current as C
HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def gate(n,e,t=3e-12):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def compare(n,a,b,scale=None,tol=3e-12):gate(n,np.max(abs(a-b))/(max(np.max(abs(a)),np.max(abs(b)),1e-100)if scale is None else scale),tol)
def epsilon(a,b,c):return 0 if len({a,b,c})<3 else(1 if(a,b,c)in[(0,1,2),(1,2,0),(2,0,1)]else -1)
# Literal Cartesians from -δH/δv: no commutator or Ward-defect call.
def gauged_vertices(k1,k2,model,raising):
 pauli=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])];u=sum(k1[i]*pauli[i]for i in range(3));v=sum(k2[i]*pauli[i]for i in range(3));D1=k1@k1+model.m_pi**2;D2=k2@k2+model.m_pi**2;sea=np.zeros((3,3,16,16),complex);flight=sea.copy()
 for b in range(3):
  for axis in range(3):
   for a in range(3):
    for c in range(3):
     e=epsilon(a,b,c)
     if e:
      # δ derivative pion-NN vertex on each line (π_source=-ig sigma.k/D).
      sea[b,axis]+=1j*model.coefficient*e*(np.kron(np.kron(pauli[axis],v)/D2,np.kron(pauli[a],pauli[c]))+np.kron(np.kron(u,pauli[axis])/D1,np.kron(pauli[c],pauli[a])))
      # δ pion gradient energy: -(∂π_a) epsilon_abc π_c.
      flight[b,axis]+=-1j*model.coefficient*e*np.kron(np.kron(u,v)/(D1*D2),k1[axis]*np.kron(pauli[a],pauli[c])+k2[axis]*np.kron(pauli[c],pauli[a]))
 sign=1 if raising else -1
 return dict(seagull=sea[0]+sign*1j*sea[1],pion_flight=flight[0]+sign*1j*flight[1])

def point(pmag,omega,Q,u):
 Q=np.array(Q,float);u=np.array(u,float);r=math.sqrt(pmag*pmag-939*omega-Q@Q/4)*u/np.linalg.norm(u);return np.array([[0.,0.,pmag],[0.,0.,-pmag],r-Q/2,-r-Q/2]),Q
points=[(117.3,6.7,[-1.7,2.9,.8],[.24,-.61,.75]),(83.1,2.8,[1.4,.7,-2.1],[-.46,.32,.83]),(128.7,7.3,[2.7,.4,-1.9],[.63,-.27,.42])]
for idx,args in enumerate(points):
 p,Q=point(*args);om=args[1];k1=p[2]-p[0];k2=p[3]-p[1];row=dict(p=p.tolist(),omega=om,Q=Q.tolist(),charge={})
 for raising in[False,True]:
  d=C.direct(k1,k2,raising=raising);ref=gauged_vertices(k1,k2,C.W.ope.Model(),raising)
  for name in['seagull','pion_flight']:compare(str((idx,raising,name))+' literal gauged-H vertex',d[name],ref[name])
  combined=C.combined(p,om,Q,raising=raising);ext=combined['external'];interaction=combined['interaction'];tot=combined['total'];defect=om*ext[0]-np.einsum('i,ijk->jk',Q,ext[1:]);cancel=np.einsum('i,ijk->jk',Q,interaction['total']);res=om*tot[0]-np.einsum('i,ijk->jk',Q,tot[1:]);compare(str((idx,raising))+' external defect canceled by sourced pion current',cancel,defect)
  scale=max(np.max(abs(om*tot[0])),np.max(abs(Q[:,None,None]*tot[1:])))
  gate(str((idx,raising))+' total static vector divergence',np.max(abs(res))/scale)
  # Each seagull/flight piece alone leaves a finite defect; relative signs matter.
  for name in['seagull','pion_flight']:
   rem=defect-np.einsum('i,ijk->jk',Q,interaction[name]);gate(str((idx,raising,name))+' singlepiece cannot complete Ward',float(np.max(abs(rem))<1e-8*np.max(abs(defect))),0)
  # Complete exchange is done coherently, piece by piece.
  swapped=C.antisymmetrized(p[[0,1,3,2]],raising=raising);inswap=C.antisymmetrized(p[[1,0,2,3]],raising=raising)
  for name in['seagull','pion_flight','total']:
   compare(str((idx,raising,name))+' outgoing exchange',np.einsum('ij,ajk->aik',C.W.ope.EXCHANGE,swapped[name]),-interaction[name])
   compare(str((idx,raising,name))+' incoming exchange',np.einsum('aij,jk->aik',inswap[name],C.W.ope.EXCHANGE),-interaction[name])
  # Nonzero sourced transverse piece, not a projector-Q times defect construction.
  jq=interaction['total'];trans=jq-Q[:,None,None]*np.einsum('i,ijk->jk',Q,jq)/(Q@Q);gate(str((idx,raising))+' sourced transverse matrix NONZERO',float(np.linalg.norm(trans)<1e-8*np.linalg.norm(jq)),0)
  rev=C.antisymmetrized(p[[2,3,0,1]],raising=not raising)
  for name in['seagull','pion_flight','total']:compare(str((idx,raising,name))+' charged adjoint reversal',rev[name],interaction[name].conj().transpose(0,2,1))
  row['charge'][str(raising)]=dict(external_defect_max=float(np.max(abs(defect))),completed_divergence_max=float(np.max(abs(res))),seagull_norm=float(np.linalg.norm(interaction['seagull'])),flight_norm=float(np.linalg.norm(interaction['pion_flight'])),transverse_norm=float(np.linalg.norm(trans)))
 rows.append(row)
# Q=0 has no coordinate singularity: the minimally sourced spatial current
# can have a finite transverse/soft value even when its divergence iszero.
p,Q=point(96.4,4.1,[0,0,0],[.13,.73,-.42]);r=C.combined(p,4.1,Q);gate('soft static vector charge zero',np.max(abs(r['total'][0]))/(C.W.ope.Model().coefficient/4.1));gate('soft pion current finite',int(not np.all(np.isfinite(r['interaction']['total']))),0)
# Pure dimensional test, no coupling or empirical normalization fit.
p,Q=point(*points[-1]);om=points[-1][1];lam=1.73;model=C.W.ope.Model(1.01,140*lam,939*lam)
for name in['seagull','pion_flight','total']:compare('MeV^-3 scale '+name,C.antisymmetrized(p*lam,model)[name]*lam**3,C.antisymmetrized(p)[name])
for name,call in [('complex',lambda:C.direct([1+1j,0,0],[0,1,0])),('NaNvector',lambda:C.direct([math.nan,0,0],[0,1,0])),('NaNgv',lambda:C.direct([1,0,0],[0,1,0],gv=math.nan)),('Boolgv',lambda:C.direct([1,0,0],[0,1,0],gv=False)),('badcharge',lambda:C.direct([1,0,0],[0,1,0],raising=math.nan)),('overflow',lambda:C.direct([1e308,0,0],[0,1,0]))]:
 try:call();bad=1
 except ValueError:bad=0
 gate('reject '+name,bad,0)
C.verify_sources();out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],points=rows,pins=C.PINS,seconds=time.monotonic()-start,scope='static minimally gauged OPE vector current only; no axial PCAC, realistic transverse completeness, retardation, weak full rate')
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','seconds']},indent=2))
