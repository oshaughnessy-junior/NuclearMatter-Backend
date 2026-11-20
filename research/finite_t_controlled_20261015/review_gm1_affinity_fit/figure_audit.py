"""Read-only consistency audit of root's fit-comparison figure data."""
from pathlib import Path
import json,hashlib,math,sys,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).parent;R=H.parent;F=R/'fit_figures';f=json.loads((F/'affinity_fit_measurements.json').read_text());d=json.loads((R/'gm1_affinity_fit/holdout_results.json').read_text());spec=importlib.util.spec_from_file_location('old_fit_readonly',R/'gm1_potential_fit/potential.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);checks=[]
def gate(n,e,t=0.):checks.append(dict(name=n,error=float(e),tolerance=t,passed=math.isfinite(e)and abs(e)<=t))
for p,sha in f['source_sha256'].items():gate('raw figure source hash '+p,float(hashlib.sha256((R/p).read_bytes()).hexdigest()!=sha))
gate('same30 state count',len(f['rows'])-len(d['rows']))
for i,(r,z)in enumerate(zip(f['rows'],d['rows'])):
 gate('same state'+str(i),float(any(r[k]!=z[k]for k in['n','T','Y','group'])))
 q=old.evaluate(r['n'],r['T'],r['Y']);ref=z['exact']['mu_n_mev']-z['exact']['mu_p_mev'];err=abs(math.expm1(((q['mu_n']-q['mu_p'])-ref)/r['T']))
 gate('independent old balance'+str(i),r['old_balance']-err,1e-12);gate('saved new balance'+str(i),r['new_balance']-z['errors']['balance'],1e-14)
 gate('old developmentpoint excluded'+str(i),float((r['n'],r['T'],r['Y'])==(.1811,.12,.1817)))
maxold=100*max(r['old_balance']for r in f['rows']);maxnew=100*max(r['new_balance']for r in f['rows'])
out=dict(passed=all(c['passed']for c in checks),checks=checks,max_old_percent=maxold,max_new_percent=maxnew,caption='Same30 new-fit blocked states for BOTH frozen potentials. Coefficients untouched. Old development failure excluded. Sampled balance-factor error, not uniform domain or actual weak emissivity.',source_figure_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in F.iterdir()if p.is_file()})
(H/'figure_audit_results.json').write_text(json.dumps(out,indent=2)+'\n');print(len(checks),out['passed'],maxold,maxnew)
