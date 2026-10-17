"""Fresh child: already accepted independent Dalitz beta recipe at new root."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('balance_review_alternate_beta',ROOT/'review_crossed_moments/reference.py');D=importlib.util.module_from_spec(spec);sys.modules[spec.name]=D;spec.loader.exec_module(D)
summary=json.loads((P/'summary.json').read_text());b=type('ExplicitMatchedBath',(),summary['balanced']['bath'])();ref=summary['balanced']['channels']['beta'];checks=[];rows=[];start=time.monotonic()
paths=[ROOT/'review_crossed_moments/reference.py',ROOT/'muon_full_current/current.py',ROOT/'review_hot_beta/REVIEW.md',ROOT/'review_pc_moments/REVIEW.md'];before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
for order,radial,orient,phi in [(20,10,8,12),(28,16,12,16)]:
 z=D.integrate(b,b.mu_n-b.mu_p-b.mu_e,order=order,radial=radial,orient=orient,phi=phi,budget=110);rows.append(z)
 ck('Dalitz incoming parent '+str(order),z['density']/(1e-7*(1-summary['kinetic']))-1,3e-8)
 for k,v in zip(D.NAMES,z['scenarios']['empty']['net']):ck('new root Dalitz-vs-spinor '+str(order)+' '+k,v/ref[k]-1,3e-6)
 print(order,z['seconds'],flush=True)
for k,a,bv in zip(D.NAMES,rows[0]['scenarios']['empty']['net'],rows[1]['scenarios']['empty']['net']):ck('Dalitz rule refinement '+k,a/bv-1,3e-6)
ck('additional dependencies unchanged',int(before!={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}),0)
(P/'alternate_beta.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],additional_frozen_hashes=before,seconds=time.monotonic()-start),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n');print(len(checks),[c for c in checks if not c['passed']],time.monotonic()-start)
