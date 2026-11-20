"""Final API qualification/provenance metadata; numerical coefficients unchanged."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
P=Path(__file__).parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('final_moment_fit_api',P/'api.py');A=importlib.util.module_from_spec(sp);sys.modules[sp.name]=A;sp.loader.exec_module(A);r=json.loads((P/'validation_results.json').read_text());checks=[]
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
for z in r['rows']:
 new=A.evaluate(z['E']);ck('unchanged numerical moments '+str((z['part'],z['E'])),all(new[k]==z['fit'][k]for k in['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']));ck('explicit signed failure status '+str(z['E']),new['measured_positive_gate_pass']is True and new['signed_energy_RT_gate_pass']is False and'detailed'in new['signed_energy_advice'])
for E in np.geomspace(1.,30.,501):
 f=A.evaluate(float(E));ck('fitted positive bounds '+str(E),f['rate']>0 and 0<f['transport_rate']<=2*f['rate']and f['outgoing_energy_moment']>0)
saved=A.QUALIFICATION_SHA;A.QUALIFICATION_SHA='0'*64
try:A.evaluate(7.);bad=True
except RuntimeError:bad=False
A.QUALIFICATION_SHA=saved;ck('qualification identity rejection',not bad)
ck('model remains exactly locked',hashlib.sha256((P/'model.json').read_bytes()).hexdigest()==json.loads((P/'FITTING_LOCK.json').read_text())['model_sha'])
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],scope='Metadata/provenance and mathematical positivity checks only; dense fit sampling is not source-error validation')
(P/'api_metadata_results.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k]for k in['passed','failed']},indent=2))
