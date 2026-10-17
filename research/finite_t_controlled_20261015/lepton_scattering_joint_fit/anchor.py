"""Preregistered fresh independent phase anchor, no coefficient changes."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time
P=Path(__file__).parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
if __name__=='__main__':
 if(P/'phase_results.json').exists():raise RuntimeError('Frozen phase evidence exists')
 V=json.loads((P/'validation.json').read_text());reg=json.loads((P/'PREREGISTRATION.json').read_text());X=load('joint_independent_phase',P/'phase_reference.py');E=reg['independent_phase_anchor']['E_MeV'];ref=next(z['fine']for z in V['rows']if z['E']==E);checks=[];rows=[]
 for n,a in reg['independent_phase_anchor']['orders']:
  row=X.integrated(E,n,a,160.,90.);rows.append(row);b=ref['total']
  for k in['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']:
   er=(row['total'][k]-b[k])/(2*b['rate'])if k=='neutrino_energy_change'else row['total'][k]/b[k]-1;checks.append(dict(name='independent D vs CM '+k,order=[n,a],error=er,gate=1e-4,passed=abs(er)<=1e-4))
  (P/'phase_partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2))
 for k in['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']:
  b=rows[-1]['total'];er=(rows[0]['total'][k]-b[k])/(2*b['rate'])if k=='neutrino_energy_change'else rows[0]['total'][k]/b[k]-1;checks.append(dict(name='D phase refinement '+k,error=er,gate=1e-4,passed=abs(er)<=1e-4))
 out=dict(rows=rows,CM=ref,checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],model_sha=hashlib.sha256((P/'model.json').read_bytes()).hexdigest());(P/'phase_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
