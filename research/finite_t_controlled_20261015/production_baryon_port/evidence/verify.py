from pathlib import Path
import sys,json,hashlib,ast,importlib.metadata,xml.etree.ElementTree as ET
from nmbackend.reference import finite_t_baryons as B
P=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=json.loads((P/'evidence/baseline_package_blobs.json').read_text());assert records['count']==71
for row in records['checks']:
 p=P/'test_mirror'/row['path'];b=p.read_bytes();actual=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest();assert actual==row['expected_git_blob'],p
assert Path(B.__file__).resolve()==(P/'test_mirror/src/nmbackend/reference/finite_t_baryons.py').resolve()
assert (P/'staged/src/nmbackend/reference/finite_t_baryons.py').read_bytes()==Path(B.__file__).read_bytes()
origins={n:str(Path(m.__file__).resolve())for n,m in sys.modules.items()if n=='nmbackend'or n.startswith('nmbackend.')if getattr(m,'__file__',None)}
assert all(str((P/'test_mirror').resolve())in p for p in origins.values())
for rel in ['src/nmbackend/reference/finite_t_baryons.py','tests/test_finite_t_baryons.py','examples/finite_t_baryons.py']:ast.parse((P/'staged'/rel).read_text(),feature_version=(3,9))
a=json.loads((P/'evidence/forward_gate.json').read_text());assert a['passed']==48 and a['failed']==[] and a['source_sha256']==sha(P/'staged/src/nmbackend/reference/finite_t_baryons.py')
r=ET.parse(P/'evidence/tests_final.xml').getroot();ss=[r]if r.tag=='testsuite'else list(r);assert sum(int(s.get('tests','0'))for s in ss)==49
assert all(int(s.get(k,'0'))==0 for s in ss for k in ['failures','errors','skipped'])
for n in ['forward_gate.stderr','tests_initial.stderr','tests_final.stderr','example.stderr']:assert (P/'evidence'/n).read_bytes()==b''
example=json.loads((P/'evidence/example.json').read_text())
for k in ['neutron','proton']:assert example[k]['particle_density']>0
out=dict(exact_baseline_package_blobs=71,actual_package_module_origins=origins,python=sys.version,packages={n:importlib.metadata.version(n)for n in ['numpy','scipy','jax','diffrax','pytest']},python39_grammar_files=3,forward_checks=48,candidate_tests=36,cold_tests=13,candidate_source_sha256=sha(Path(B.__file__)))
(P/'evidence/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
