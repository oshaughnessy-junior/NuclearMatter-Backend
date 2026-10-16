from pathlib import Path
import sys,json,hashlib,ast,xml.etree.ElementTree as ET,importlib.metadata,difflib
P=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference import finite_t_rmf as R
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
base=json.loads((P/'evidence/baseline_package_blobs.json').read_text());assert all(blob((P/'test_mirror'/r['path']).read_bytes())==r['expected_git_blob']for r in base['checks'])
paths=[p for p in (P/'staged').rglob('*')if p.is_file()];assert len(paths)==9
for p in paths:
 assert p.read_bytes()==(P/'test_mirror'/p.relative_to(P/'staged')).read_bytes()
 if p.suffix=='.py':ast.parse(p.read_text(),feature_version=(3,9))
origins={n:str(Path(m.__file__).resolve())for n,m in sys.modules.items()if (n=='nmbackend'or n.startswith('nmbackend.'))and getattr(m,'__file__',None)};assert all(str((P/'test_mirror').resolve())in v for v in origins.values());assert R.__all__==['SolveControls','UniformRMFState','state_density','diagnose_density']
assert R.UniformRMFState.__dataclass_params__.frozen and R.SolveControls.__dataclass_params__.frozen
batches={}
for name,expected in [('final_basic',23),('final_ladders',2),('final_remaining_cold',20)]:
 root=ET.parse(P/'evidence'/f'{name}.xml').getroot();ss=list(root)if root.tag=='testsuites'else[root];total=sum(int(s.get('tests',0))for s in ss);assert total==expected,(name,total);assert not any(int(s.get(k,0))for s in ss for k in ['failures','errors','skipped']);batches[name]=total
for name in ['final_basic','final_ladders','final_remaining_cold','final_forward','response_probe','example']:
 assert (P/'evidence'/f'{name}.stderr').read_bytes()==b''
f=json.loads((P/'evidence/forward_results.json').read_text());assert f['passed']==65 and not f['failed']
probe=json.loads((P/'evidence/response_probe.json').read_text());assert all(c['passed']for c in probe['checks'])
ex=json.loads((P/'evidence/example.json').read_text());assert len(ex['states'])==2 and all(s['locally_admitted']for s in ex['states']);assert not ex['diagnostic_only_unadmitted']['locally_admitted']
for k,h in json.loads((P/'evidence/baseline_research_pin.json').read_text()).items():assert sha(P.parent/k)==h,k
baseline=(P/'evidence/baseline_README.md').read_text();new=(P/'staged/README.md').read_text();assert new.startswith(baseline);(P/'evidence/README.diff').write_text(''.join(difflib.unified_diff(baseline.splitlines(True),new.splitlines(True),fromfile='README baseline',tofile='README candidate')))
out=dict(exact_baseline_blobs=len(base['checks']),candidate_paths={str(p.relative_to(P/'staged')):sha(p)for p in paths},actual_package_origins=origins,tests=batches,unique_candidate_tests=32,unchanged_cold_unit_tests=13,forward_field_checks=65,extra_raw_response_checks=3,python39_grammar_files=sum(p.suffix=='.py'for p in paths),python=sys.version,versions={n:importlib.metadata.version(n)for n in ['numpy','scipy','jax','diffrax','pytest']},external_native_executed=False,full_CI=False,minimum_runtime_executed=False)
(P/'evidence/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
