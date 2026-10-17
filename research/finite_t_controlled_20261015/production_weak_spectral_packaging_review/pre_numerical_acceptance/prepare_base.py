"""Read-only-source base assembly; never inspect/import mutable StageB candidate."""
from pathlib import Path
import json,hashlib,shutil,ast,datetime,sys,importlib.metadata
P=Path(__file__).resolve().parent;B=P/'package_base';S=P.parent/'production_weak_packaging_review';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert not B.exists()
records=json.loads((S/'PREPARATION.json').read_text());assert records['base']=='ffb5cc72949f2e5401251847aef4c9c7a98dde78'and len(records['baseline_blobs'])==71
basepins={}
for r in records['baseline_blobs']:
 p=S/'package_copy'/r['path'];raw=p.read_bytes();assert blob(raw)==r['blob'];q=B/r['path'];q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);basepins[r['path']]=dict(git_blob=r['blob'],sha256=sha(q),provenance='71exactffbbaseline; PR54commitdiff does not modify this path')
r=json.loads((P/'PR54_SOURCE_READ.json').read_text());commit=json.loads((P/'PR54_COMMIT_READ.json').read_text())['structuredContent']['commit'];assert commit['sha']=='c17472b8a84655380ab64973ef883a6cd5fb2346';assert sorted(r['paths'])==sorted(x['filename']for x in commit['files'])and len(r['paths'])==11
publication=json.loads((P.parent/'production_weak_publication/hashes.json').read_text());stageA=json.loads((P.parent/'production_weak_port/CANDIDATE_MANIFEST.json').read_text());assert stageA['files']['src/nmbackend/reference/_weak_quadrature.py']=='2d9a016f8ab629209856c40ac2ba4dac132280be1cfba82883586dd66decfd69'
for path,item in zip(r['paths'],r['results']):
 assert item['status']=='fulfilled'and not item['value'].get('isError');d=item['value']['structuredContent'];assert d['encoding']=='utf-8';raw=d['content'].encode();assert blob(raw)==d['sha']==publication[path];assert hashlib.sha256(raw).hexdigest()==stageA['files'][path];q=B/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);basepins[path]=dict(git_blob=d['sha'],sha256=sha(q),provenance='fresh exactPR54head contents read')
metadata=json.loads((P/'PR54_METADATA_READ.json').read_text())
for path,item in zip(['pyproject.toml','LICENSE'],metadata):
 assert item['status']=='fulfilled'and not item['value'].get('isError');d=item['value']['structuredContent'];raw=d['content'].encode();assert blob(raw)==d['sha'];q=B/path;q.write_bytes(raw);assert q.read_bytes()==(S/'package_copy'/path).read_bytes();basepins[path]=dict(git_blob=d['sha'],sha256=sha(q),provenance='fresh exactPR54head contents read, unchangedbaseline')
for name in ['finite_t_leptons.py','finite_t_baryons.py','finite_t_rmf.py','_weak_spectral_common.py','_weak_spectral_capture.py','_weak_spectral_decay.py']:assert not(B/'src/nmbackend/reference'/name).exists()
for path in ['src/nmbackend/reference/weak_reference.py','src/nmbackend/reference/_weak_current.py','src/nmbackend/reference/_weak_two_body.py','src/nmbackend/reference/_weak_decay.py','src/nmbackend/reference/_weak_quadrature.py','tests/test_weak_reference_current.py','tests/test_weak_reference_moments.py']:ast.parse((B/path).read_text(),feature_version=(3,9))
# Exact existing cold tests, separately pinned rather than a package stub/loader.
for p in sorted((S/'cold_tests').glob('test_*.py')):
 q=B/'tests'/p.name;q.write_bytes(p.read_bytes());basepins[str(q.relative_to(B))]=dict(sha256=sha(q),provenance='unchanged exact coldtests from accepted StageApackaging; COLD_SOURCE_PIN checked');assert sha(q)==json.loads((S/'COLD_SOURCE_PIN.json').read_text())[p.name]
(P/'BASE_PIN.json').write_text(json.dumps(dict(head=commit['sha'],baseline='ffb5cc72949f2e5401251847aef4c9c7a98dde78',files=basepins),indent=2)+'\n')
result=dict(status='FROZEN_BASE_PREPARATION_ONLY_WAITING_ROOT_DIFFERENT_NUMERICAL_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=commit['sha'],exact_baseline_blobs=71,exact_PR54_overlay_reads=11,exact_pyproject_license_reads=2,base_files_pinned=len(basepins),candidate_read=False,candidate_copied=False,candidate_imports=False,wheel_build=False,physical_calls=0,minimum_runtime_execution=False,python=sys.version,versions={n:importlib.metadata.version(n)for n in ['numpy','scipy','jax','diffrax','setuptools','wheel']},base_pin_sha256=sha(P/'BASE_PIN.json'))
(P/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
