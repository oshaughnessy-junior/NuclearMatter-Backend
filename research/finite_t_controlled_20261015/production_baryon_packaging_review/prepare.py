from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent; A=P.parent/'production_baryon_port'; B=P/'package_copy'; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); blob=lambda b:hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
assert sha(A/'CANDIDATE_MANIFEST.json')=='199716b691d88a4fc9b83ed7f8e8b48e36fbff16fcc9b1e8684d18d0922fc523'
manifest=json.loads((A/'CANDIDATE_MANIFEST.json').read_text()); pins={str(f.relative_to(A)):sha(f) for f in A.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
shutil.copytree(A/'test_mirror/src',B/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc')); shutil.copy2(A/'test_mirror/pyproject.toml',B/'pyproject.toml')
records=json.loads((A/'evidence/baseline_package_blobs.json').read_text());checks=[]
for r in records['checks']:
 actual=blob((B/r['path']).read_bytes()); assert actual==r['expected_git_blob']; checks.append(dict(path=r['path'],blob=actual))
# License exact baseline evidence already pinned in independent lepton packaging audit.
T=P.parent/'production_packaging_review'; provenance=json.loads((T/'PREPARATION.json').read_text()); assert sha(T/'package_copy/LICENSE')==provenance['license_sha256'];shutil.copy2(T/'package_copy/LICENSE',B/'LICENSE')
pyrec=json.loads((P.parent/'production_lepton_port/evidence/github_source_records.json').read_text()); pr=next(r for r in pyrec if r['path']=='pyproject.toml');assert blob((B/'pyproject.toml').read_bytes())==pr['sha']
for path,h in manifest['files'].items():
 assert sha(A/'staged'/path)==h; target=B/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/'staged'/path,target)
(P/'BEFORE_PIN.json').write_text(json.dumps(pins,indent=2)+'\n'); (P/'PREPARATION.json').write_text(json.dumps(dict(base=records['base'],baseline_blobs=checks,candidate_paths=manifest['files'],license_sha256=provenance['license_sha256'],license_source='Read-only prior packaging baseline license; prior exact git blob and unchanged-head evidence retained',pyproject_git_blob=pr['sha']),indent=2)+'\n');print(len(checks),'exact baseline blobs and',len(manifest['files']),'candidate overlays')
