"""Compare frozen forward data only; no independent-oracle implementation read."""
from pathlib import Path
import json,hashlib,math
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent
a=json.loads((P/'results.json').read_text());path=ROOT/'relativistic_delta_oracle/results.json';b=json.loads(path.read_text());checks=[];rows=[]
def complex_array(d):return np.array(d['real'])+1j*np.array(d['imag'])
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
for own,oracle in zip(a['rows'][:3],b['rows']):
 x,y=own['runs'][-1],oracle['runs'][-1];p=oracle['point']
 for key,other in[('T',p['T']),('mass_a',p['ma']),('mass_b',p['mb']),('mu_a',p['mua']),('mu_b',p['mub']),('weak_mass',p['Mweak'])]:gate(own['label']+' exact input '+key,own['state'][key]-other,0.)
 for key,other in[('Enu',p['Enu']),('El',p['El']),('cosine',p['cosine']),('lepton_mass',p['ml']),('mu_lepton',p['mul']),('antineutrino',p['anti'])]:gate(own['label']+' exact lepton input '+key,float(own['inputs'].get(key,False))-float(other),0.)
 for key in('j','chi'):gate(own['label']+' forward '+key,x[key]['value']/y[key]-1,3e-8)
 W=complex_array(y['W'])
 for index,key in enumerate(('absorption','emission')):
  tensor=complex_array(x['response'][key]['tensor']);error=float(np.max(abs(tensor-W[index]))/np.max(abs(W[index])))
  gate(own['label']+' all16 complex '+key,error,3e-8)
 L=complex_array(x['lepton_tensor']);Lor=complex_array(y['lepton_tensor'])
 gate(own['label']+' all16 complex lepton',np.max(abs(L-Lor))/np.max(abs(Lor)),3e-12)
 rows.append(dict(label=own['label'],author_j=x['j']['value'],author_chi=x['chi']['value'],oracle_j=y['j'],oracle_chi=y['chi']))
out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,scope='forward data agreement only, independent reviews remain required',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),P/'tensor.py',P/'results.json',path]})
(P/'oracle_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],max_data_error=max(abs(c['error'])for c in checks))))
if not out['passed']:raise SystemExit(1)
