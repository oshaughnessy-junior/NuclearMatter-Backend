"""Copied-fixture source-pin guards; never changes frozen author files."""
from pathlib import Path
import hashlib,importlib.util,sys,json
P=Path(__file__).resolve().parent;A=P.parent/'lepton_scattering_joint_fit'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
before={x.name:h(x) for x in [A/'api.py',A/'model.json',A/'reference.py']}
(P/'copied_api.py').write_bytes((A/'api.py').read_bytes());(P/'model.json').write_bytes((A/'model.json').read_bytes())
api=load('review_joint_copied_api',P/'copied_api.py');checks=[]
checks.append(dict(name='copied pristine model matches literal pin',passed=api.model()['version']=='joint-lepton-moment-2'))
(P/'model.json').write_bytes((A/'model.json').read_bytes()+b' ')
try:api.evaluate(8.1137);checks.append(dict(name='stale model rejects before physical invocation',passed=False))
except RuntimeError as e:checks.append(dict(name='stale model rejects before physical invocation',passed=str(e)=='Frozen model changed'))
(P/'stale_model.json').write_bytes((P/'model.json').read_bytes());(P/'model.json').write_bytes((A/'model.json').read_bytes())
(P/'reference.py').write_bytes((A/'reference.py').read_bytes()+b'\n# intentionally stale reviewer fixture\n')
try:api.evaluate(8.1137);checks.append(dict(name='stale detailed adapter rejects before physical invocation',passed=False))
except RuntimeError as e:checks.append(dict(name='stale detailed adapter rejects before physical invocation',passed=str(e)=='Actual fallback adapter changed'))
# Pristine source import only, no expensive moment call. Record real module origins.
ref=load('review_joint_origin_only_reference',A/'reference.py')
origins={n:str(Path(sys.modules[n].__file__).resolve()) for n in ['fit_physical_scattering','pair','events']}
expect={'fit_physical_scattering':P.parent/'neutrino_lepton_scattering/scatter.py','pair':P.parent/'leptonic_pair_annihilation/pair.py','events':P.parent/'relativistic_capture_oracle/events.py'}
checks.append(dict(name='fresh-process actual function module origins match pinned files',passed=all(origins[n]==str(v.resolve()) for n,v in expect.items())))
checks.append(dict(name='frozen api/model/reference bytes unchanged',passed=before=={x.name:h(x) for x in [A/'api.py',A/'model.json',A/'reference.py']}))
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],origins=origins,no_physical_rerun=True)
(P/'pin_guard_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
