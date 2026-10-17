from pathlib import Path
import sys,importlib.util,json,math,hashlib
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('capture_fit_exact_ec_physics',R/'hot_electron_capture_moments/electron_capture.py');C=importlib.util.module_from_spec(s);sys.modules[s.name]=C;s.loader.exec_module(C)
lock=json.loads((P/'LOCK.json').read_text());rows=[]
for group in ['training','heldout']:
 for state in lock[group]:
  b,st=C.matched(**state);eta=dict(e=b.mu_e/b.T,n=(b.mu_n-C.MN)/b.T,p=(b.mu_p-C.MP)/b.T)
  admitted=.2<=b.T<=4 and max(eta['n'],eta['p'])<=-5 and abs(eta['e'])<=3
  rows.append(dict(group=group,state=state,bath=vars(b),etas=eta,EC_existing_domain_admitted=admitted,density_residuals=dict(n=st['neutron']['density']/(state['nB']*(1-state['Ye']))-1,p=st['proton']['density']/(state['nB']*state['Ye'])-1,e=st['electron']['net_density']/(state['nB']*state['Ye'])-1)))
out=dict(rows=rows,all_admitted=all(r['EC_existing_domain_admitted']for r in rows),blocked=[r for r in rows if not r['EC_existing_domain_admitted']],lock_sha256=hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest(),scientific_source_calls=0)
(P/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(admitted=out['all_admitted'],blocked=out['blocked']),indent=2))
