"""Static package assembly ONLY; do not import/build/execute candidate."""
from pathlib import Path
import shutil,json,hashlib,ast,difflib,datetime
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_port';B=P/'package_copy';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert not B.exists()
assert sha(A/'CANDIDATE_MANIFEST.json')=='80f0b4207003353b8bf59f1357c28baedbf85ccb06d0a85b9289bceb798a174e'
assert sha(A/'MANIFEST.json')=='347ee290afa7eb4d177b92c2ee65056c90cea2a388d13d74ac4e5f5e8f8c6cd7'
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert len(m['files'])==11
pins={str(p.relative_to(A)):sha(p)for p in A.rglob('*')if p.is_file()and '__pycache__'not in p.parts}
for rel,h in json.loads((A/'MANIFEST.json').read_text())['sha256'].items():assert sha(A/rel)==h
records=json.loads((A/'evidence/baseline_package_blobs.json').read_text());assert records['base']==m['base_commit']=='ffb5cc72949f2e5401251847aef4c9c7a98dde78';assert len(records['checks'])==71
checks=[]
for r in records['checks']:
 source=A/'test_mirror'/r['path'];assert blob(source.read_bytes())==r['expected_git_blob'];dest=B/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);checks.append(dict(path=r['path'],blob=r['expected_git_blob']))
T=P.parent/'production_packaging_review';provenance=json.loads((T/'PREPARATION.json').read_text());assert sha(T/'package_copy/LICENSE')==provenance['license_sha256'];shutil.copy2(T/'package_copy/LICENSE',B/'LICENSE')
pyrec=json.loads((P.parent/'production_lepton_port/evidence/github_source_records.json').read_text());pr=next(r for r in pyrec if r['path']=='pyproject.toml');assert blob((A/'test_mirror/pyproject.toml').read_bytes())==pr['sha'];shutil.copy2(A/'test_mirror/pyproject.toml',B/'pyproject.toml')
static=[]
for path,h in m['files'].items():
 assert sha(A/'staged'/path)==h;dest=B/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/'staged'/path,dest)
 if path.endswith('.py'):
  text=dest.read_text();tree=ast.parse(text,feature_version=(3,9));static.append(dict(path=path,python39_grammar=True,imports=[ast.unparse(n)for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))],call_targets=sorted({ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)})))
assert len(static)==8
for name in ['finite_t_rmf.py','finite_t_leptons.py','finite_t_baryons.py']:assert not(B/'src/nmbackend/reference'/name).exists()
old=(A/'evidence/baseline_README.md').read_text();new=(B/'README.md').read_text();assert new.startswith(old)
(P/'README.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='ffb baseline',tofile='candidate')))
(P/'BEFORE_PIN.json').write_text(json.dumps(pins,indent=2)+'\n')
result=dict(status='REPAIRED_PACKAGE_PREPARED_NUMERICAL_ACCEPTANCE_RECEIVED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base=records['base'],baseline_blobs=checks,candidate_paths=m['files'],license_sha256=provenance['license_sha256'],pyproject_git_blob=pr['sha'],static_ast=static,candidate_paths_count=11,source_modules=5,all_candidate_files_unchanged=pins,actual_candidate_imports=False,wheel_build=False,physical_calls=False,minimum_runtime_executed=False,source_scope='10new files plus README additive section; no unmerged RMF/lepton/baryon modules or baseline Python edits')
assert all(sha(A/rel)==h for rel,h in pins.items())
(P/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n');print('PREPARED71baseline+11overlays;8Python3.9AST parses; no candidate imports/builds/calls; ROOT_SIGNAL_RECEIVED')
