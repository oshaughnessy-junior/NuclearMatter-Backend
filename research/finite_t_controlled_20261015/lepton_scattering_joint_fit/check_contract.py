"""Pure API contract/math checks plus one independently requested fallback."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib
import numpy as np
P=Path(__file__).parent
if(P/'api_results.json').exists():raise RuntimeError('Refusing to overwrite frozen API evidence')
s=importlib.util.spec_from_file_location('joint_contract_api',P/'api.py');A=importlib.util.module_from_spec(s);sys.modules[s.name]=A;s.loader.exec_module(A)
checks=[]
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
for E in np.geomspace(1,30,301):
 z=A.evaluate(float(E),method='diagnostic_fit');ck('positive and angular bounds '+str(E),z['rate']>0 and 0<z['transport_rate']<=2*z['rate']and z['outgoing_energy_moment']>0);ck('joint ledger '+str(E),abs(z['outgoing_energy_moment']-E*z['rate']-z['neutrino_energy_change'])<1e-12*z['rate']*max(E,2))
for x in[True,False,float('nan'),float('inf'),0.,.99,30.01,-1.,'2',None]:
 try:A.evaluate(x);ok=False
 except ValueError:ok=True
 ck('reject input '+str(x),ok)
for kw in[dict(T=True),dict(mu_e=False),dict(T=2.00001),dict(mu_e=A.MU+1e-12),dict(T=float('nan')),dict(flavor='anti_nu_e'),dict(targets=('e-',)),dict(outgoing='FD'),dict(method='fit'),dict(targets=['e-','e+'])]:
 try:A.evaluate(2.,**kw);ok=False
 except ValueError:ok=True
 ck('reject closure '+str(kw),ok)
# A NEW explicit call outside cancellation proves detailed mode is not merely a stored validation-row lookup.
z=A.evaluate(5.731,method='detailed');ck('explicit fresh detailed E5.731',z['fallback']and z['method']=='actual detailed CM quadrature'and z['E']==5.731 and z['controls']==[32,40,40,48,80.]);ck('actual detailed energy accounting',abs(z['outgoing_energy_moment']-5.731*z['rate']-z['neutrino_energy_change'])<=1e-10*z['rate']);out=dict(checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],fresh_fallback=z,model_sha=hashlib.sha256((P/'model.json').read_bytes()).hexdigest());(P/'api_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
