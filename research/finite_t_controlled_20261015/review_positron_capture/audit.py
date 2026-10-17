from pathlib import Path
import sys,json,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path[:0]=[str(R/'positron_capture'),str(R/'relativistic_delta_oracle')]
import pc_kernel as pc,oracle
checks=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
b,meta=pc.matched(1.7,2.3e-7,.27)
for name,fn in [('nanEnu',lambda:pc.differential(b,math.nan,3,.2)),('negativeEnu',lambda:pc.differential(b,-1,3,.2)),('nanEl',lambda:pc.differential(b,5,math.nan,.2)),('badcos',lambda:pc.differential(b,5,3,1.01)),('Elrest',lambda:pc.differential(b,5,pc.ME,.2)),('boolinner',lambda:pc.differential(b,5,3,.2,energy_order=True)),('nantail',lambda:pc.differential(b,5,3,.2,tail=math.nan)),('negativeTail',lambda:pc.differential(b,5,3,.2,tail=-2)),('boolouter',lambda:pc.integrate(b,5,lepton_order=True)),('badoutertail',lambda:pc.integrate(b,5,outer_tail=61)),('nanbudget',lambda:pc.integrate(b,5,budget_seconds=math.nan)),('invalidYe',lambda:pc.matched(1.7,2.3e-7,1.2))]:
 try:fn();gate('reject '+name,1,0)
 except ValueError:gate('reject '+name,0,0)
# Exact rest-target threshold from invariant s, not a finiteT support cut.
th=((pc.crossed.MN+pc.ME)**2-pc.crossed.MP**2)/(2*pc.crossed.MP)
gate('physical invariant threshold',th-1.8060662080187608,3e-11)
# Independent current recurrence: source-independent spinor implementation vs literal adapter.
for w,q in [(.37,4.8),(-2.7,15.1),(.91,1.3)]:
 q4=np.array([w,0.,0.,q]);literal=pc.reciprocal_vertices(q4,pc.crossed.MW,pc.K.C.FormFactors());ind=np.array([oracle.adj(v)for v in oracle.currents(-q4,pc.crossed.MW)])
 gate('independent reciprocal vertex',np.max(abs(literal-ind)),3e-13)
# Charge and electron-lepton ledger for forward n+e+->p+anti.
initial=dict(n=1,p=0,positron=1,antinu=0);final=dict(n=0,p=1,positron=0,antinu=1)
for name,q in [('electric',dict(n=0,p=1,positron=1,antinu=0)),('Le',dict(n=0,p=0,positron=-1,antinu=-1)),('baryon',dict(n=1,p=1,positron=0,antinu=0))]:gate('reaction '+name,sum((final[k]-initial[k])*q[k]for k in initial),0)
# Original author failed gates are intentionally retained, not silently recounted as successes.
f=json.loads((R/'positron_capture/results.json').read_text());failed=[c for c in f['checks']if not c['passed']];gate('six original coarse failures retained',len(failed)-6,0)
(P/'audit_results.json').write_text(json.dumps(dict(checks=checks,passed=all(c['passed']for c in checks),rest_threshold=th,original_coarse_failed_checks=failed,primary_source='https://arxiv.org/html/2006.12051v2',source_sections='26–29 current and spin, 47a–h invariants; free specialization only'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
