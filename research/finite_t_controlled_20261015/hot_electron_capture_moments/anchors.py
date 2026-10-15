"""Recompute independent literal reciprocal/current compatibility anchors."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util
import numpy as np
import electron_capture as C
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;bath,state=C.matched();rows=[]
for ee,c in [(1.31,-.7),(2.,.2),(9.,-.33),(27.,.61)]:
 p=np.array([[C.MP,0,0,0.]]);l=np.array([[ee,0,0,math.sqrt(ee*ee-C.ME*C.ME)]]);total=p+l;s=C.dot(total,total);rt=np.sqrt(s);pf=(s-C.MN*C.MN)/(2*rt);beta=total[:,1:]/total[:,0,None]
 k=C.E.boost(np.column_stack([pf,pf*math.sqrt(1-c*c),pf*0,pf*c]),beta);n=p+l-k
 a=C.scalar_amplitude(k,n,l,p)[0];b=C.spinor_amplitude(k,n,l,p)[0]
 ref=C.E.amplitude(C.E.Bath(bath.T,C.MN,C.MP,bath.mu_n,bath.mu_p,bath.mu_e,ml=C.ME,Mweak=C.MW),k,n,l,p)[0]/2
 rows.append(dict(Ee=ee,cosine=c,scalar=float(a),literal_reciprocal=float(b),capture_time_reverse=float(ref),relative_scalar=float(a/b-1),relative_reciprocity=float(ref/b-1)))
(HERE/'initial_amplitude_anchors.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
def load(name,file):
 sp=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
P=load('EC_PC_compatibility','positron_capture/pc_kernel.py');B=load('EC_beta_compatibility','relativistic_crossed_contract/beta.py');pc=json.loads((ROOT/'positron_capture_moments/MATCHED_STATE.json').read_text());checks=[]
checks.append(dict(name='exact common physical bath and EOS metadata',passed=state==pc))
checks.append(dict(name='same physical MNMPme bareweakavg and GF Vud',passed=(C.MN,C.MP,C.ME,C.MW,C.E.GF,C.E.VUD)==(P.crossed.MN,P.crossed.MP,P.ME,P.crossed.MW,P.V.GF,P.V.VUD)==(B.MN,B.MP,B.ME,(B.MN+B.MP)/2,B.GF,B.VUD)))
for q in [np.array([-2.4,1.1,-.7,3.2]),np.array([.3,2.1,.4,-1.7]),np.array([-6.,-4.,1.,3.])]:
 ff=P.K.C.FormFactors().values(float(q[0]*q[0]-q[1:]@q[1:]),C.MW);pieces=P.K.C.vertices(q,C.MW,ff);source=np.array([sum(pieces[name][i]for name in pieces)for i in range(4)]);a=C.E.vertices(q[None,:],C.MW)[0];error=float(np.max(abs(a-source))/max(1.,np.max(abs(a))))
 checks.append(dict(name='full current identity '+str(q.tolist()),passed=error<3e-13,error=error,tolerance=3e-13))
prev=json.loads((HERE/'INITIAL_UPSTREAM.json').read_text());checks.append(dict(name='every previously frozen upstream hash unchanged',passed=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in prev.items())))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],state_compatible_for_future_reviewed_join=all(x['passed']for x in checks),join_performed=False,notes='Shared fulltreecurrent and actualmatched bath only; radiation scenarios must also match before source sum.')
(HERE/'compatibility_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed'],amplitude_anchors=len(rows))))
