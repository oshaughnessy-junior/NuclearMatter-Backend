"""New coarse-grid time halves test spatial differences' time contamination."""
from pathlib import Path
import sys,math,time,json,hashlib
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
start=time.monotonic();d=json.loads((H/'results.json').read_text());p=slab.Physics();checks=[];rows=[]
for path,sha in d['initial_hashes'].items():
 if hashlib.sha256((R/path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen physical source mismatch; no results may be reused')
initial=next(r['initialization']['compatible_W']for r in d['rows']if'initialization'in r);saved={(r['run']['controls']['N'],r['run']['controls']['steps']):r['run']for r in d['rows']if'run'in r}
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def project(v,N):return np.array(v).reshape(N,len(v)//N).mean(axis=1)
def norms(v):return dict(L1=float(np.mean(abs(v))),L2=float(np.sqrt(np.mean(v*v))),Linf=float(np.max(abs(v))))
for N in[4,8]:
 W=np.array(initial[str(N)]);theta=np.array([p.inverse(w,1+.04*math.cos(math.pi*(i+.5)/N),(.95,1.05))for i,w in enumerate(W)]);r=slab.evolve(p,theta,.0015,32,budget_seconds=90.)
 r['final_W']=[p.W(t)for t in r['theta']];saved[(N,32)]=r;rows.append(dict(run=r));gate('coarse half-time energy '+str(N),max(abs(h['material_energy']-r['initial_energy'])for h in r['history']),5e-10)
for N in[4,8]:
 difference16=np.array(saved[(N,16)]['final_W'])-project(saved[(2*N,16)]['final_W'],N)
 difference32=np.array(saved[(N,32)]['final_W'])-project(saved[(2*N,32)]['final_W'],N)
 old=norms(difference16);new=norms(difference32);change=norms(difference16-difference32)
 for norm in['L1','L2','Linf']:gate('spatial residual unchanged under time-halving '+str((N,norm)),change[norm]/old[norm],.005)
 rows.append(dict(time_separation=dict(N=N,spatial_difference_steps16=old,spatial_difference_steps32=new,change=change,fraction={k:change[k]/old[k]for k in old})))
out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,original_results_sha256=hashlib.sha256((H/'results.json').read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-start)
(H/'time_separation_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],runtime=out['runtime_seconds'])))
