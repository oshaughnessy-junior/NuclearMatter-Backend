from pathlib import Path
import sys,json,hashlib,ast,xml.etree.ElementTree as ET,importlib.metadata,difflib
P=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
base=json.loads((P/'evidence/baseline_package_blobs.json').read_text());assert all(blob((P/'test_mirror'/r['path']).read_bytes())==r['expected_git_blob']for r in base['checks'])
paths=[q for q in(P/'staged').rglob('*')if q.is_file()];assert len(paths)==11
for q in paths:
 assert q.read_bytes()==(P/'test_mirror'/q.relative_to(P/'staged')).read_bytes()
 if q.suffix=='.py':ast.parse(q.read_text(),feature_version=(3,9))
modules={n:str(Path(m.__file__).resolve())for n,m in sys.modules.items()if(n=='nmbackend'or n.startswith('nmbackend.'))and getattr(m,'__file__',None)};assert all(str((P/'test_mirror').resolve())in x for x in modules.values())
assert not any(n in sys.modules for n in['beta','kernel','current','vacuum_cc','lepton_reference','study'])
assert not any(n in sys.modules for n in['nmbackend.reference.finite_t_leptons','nmbackend.reference.finite_t_baryons','nmbackend.reference.finite_t_rmf'])
for name in['FreeElectronBath','Radiation','Controls','Current','MomentResult','PositiveQuantity','SignedQuantity']:assert getattr(W,name).__dataclass_params__.frozen
for q in(P/'staged/src/nmbackend/reference').glob('*weak*.py'):
 text=q.read_text();assert not any(x in text for x in['importlib','sys.path','sys.modules','Path(','open(','read_text','read_bytes','json.load','loadtxt','eval(','exec('])
 # Only package-relative imports and declared stdlib/NumPy/SciPy.
 for n in ast.walk(ast.parse(text)):
  if isinstance(n,ast.Import):assert all(a.name.split('.')[0]in['numpy','math','numbers','time']for a in n.names)
  if isinstance(n,ast.ImportFrom)and n.level==0:assert n.module.split('.')[0]in['__future__','dataclasses','numpy','scipy']
  assert not(isinstance(n,ast.ImportFrom)and n.module and 'log_expit'in[a.name for a in n.names])
root=ET.parse(P/'evidence/final_tests.xml').getroot();ss=list(root)if root.tag=='testsuites'else[root];count=sum(int(s.get('tests',0))for s in ss);assert count==53,(count,53);assert not any(int(s.get(k,0))for s in ss for k in['failures','errors','skipped'])
assert(P/'evidence/final_tests.stderr').read_bytes()==b''
assessment=json.loads((P/'evidence/ASSESSMENT.json').read_text());assert assessment['passed']==1505 and assessment['failed']==0
anchors=json.loads((P/'evidence/anchors.json').read_text());assert anchors['passed']==13 and anchors['failed']==0
fingerprint={q.name:sha(q)for q in(P/'test_mirror/src/nmbackend/reference').glob('*weak*.py')}
for q in(P/'evidence').glob('candidate_*.json'):
 d=json.loads(q.read_text());assert d['complete']is True and d['candidate_sources']==fingerprint
pins=json.loads((P/'evidence/SOURCE_PIN.json').read_text());assert all(sha(P.parent/k)==h for k,h in pins.items())
readme=(P/'evidence/baseline_README.md').read_text();new=(P/'staged/README.md').read_text();assert new.startswith(readme);(P/'evidence/README.diff').write_text(''.join(difflib.unified_diff(readme.splitlines(True),new.splitlines(True))))
out=dict(exact_baseline_package_blobs=len(base['checks']),candidate_paths={str(q.relative_to(P/'staged')):sha(q)for q in paths},actual_package_origins=modules,pytest_tests=count,new_tests=40,unchanged_cold_unit_tests=13,related_raw_assessment_checks=1505,raw_anchor_checks=13,source_files_unchanged=len(pins),candidate_sources=fingerprint,python39_grammar='PASS static only',versions={n:importlib.metadata.version(n)for n in['numpy','scipy','jax','pytest']},python=sys.version,minimum_runtime_executed=False,normal_CI=False,packaging_review=False,independent_review=False,physical_model_error_estimated=False)
(P/'evidence/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
