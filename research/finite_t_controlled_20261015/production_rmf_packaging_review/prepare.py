from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'production_rmf_port';B=P/'package_copy';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert sha(A/'CANDIDATE_MANIFEST.json')=='185afee22cda5164e62eaff36911125bf7ffc9907427dd552f45cfe322aa55e3'
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert len(m['files'])==9
pins={str(p.relative_to(A)):sha(p)for p in A.rglob('*')if p.is_file()and '__pycache__'not in p.parts}
R=P.parent/'review_production_rmf_port'
for f,h in [('REVIEW.md','175f5dda33cfe67dd6ef3ec9b2a0bd6076cca79f7274b189cd76d34aa952d14f'),('MANIFEST.json','12fbccf25b035c141dcc6d1ae0e700ee8afac7e3d4785456b7be5e810011ee28'),('CHECKPOINT.json','937019dc454f89052f8921c1d34fae1cdca1e9b545d99ed5b6ba5eb124c52714')]:assert sha(R/f)==h
rpins={str(p.relative_to(R)):sha(p)for p in R.rglob('*')if p.is_file()and '__pycache__'not in p.parts}
records=json.loads((A/'evidence/baseline_package_blobs.json').read_text());assert records['base']==m['base_commit']=='ffb5cc72949f2e5401251847aef4c9c7a98dde78';assert len(records['checks'])==71
checks=[]
for r in records['checks']:
 source=A/'test_mirror'/r['path'];assert blob(source.read_bytes())==r['expected_git_blob'];dest=B/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);checks.append(dict(path=r['path'],blob=r['expected_git_blob']))
T=P.parent/'production_packaging_review';provenance=json.loads((T/'PREPARATION.json').read_text());assert sha(T/'package_copy/LICENSE')==provenance['license_sha256'];shutil.copy2(T/'package_copy/LICENSE',B/'LICENSE')
pyrec=json.loads((P.parent/'production_lepton_port/evidence/github_source_records.json').read_text());pr=next(r for r in pyrec if r['path']=='pyproject.toml');assert blob((A/'test_mirror/pyproject.toml').read_bytes())==pr['sha'];shutil.copy2(A/'test_mirror/pyproject.toml',B/'pyproject.toml')
for path,h in m['files'].items():
 assert sha(A/'staged'/path)==h;dest=B/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/'staged'/path,dest)
assert not(B/'src/nmbackend/reference/finite_t_leptons.py').exists()and not(B/'src/nmbackend/reference/finite_t_baryons.py').exists()
(P/'BEFORE_PIN.json').write_text(json.dumps(dict(author=pins,review=rpins),indent=2)+'\n')
(P/'PREPARATION.json').write_text(json.dumps(dict(base=records['base'],baseline_blobs=checks,candidate_paths=m['files'],license_sha256=provenance['license_sha256'],pyproject_git_blob=pr['sha'],instruction_evidence={str(p.relative_to(A)):sha(p)for p in [A/'evidence/instructions_readback.json',A/'evidence/data_instructions_readback.json']},exact_scope='8 new files, README additive section; no baseline Python changes, no unmerged lepton/baryon overlay'),indent=2)+'\n')
print('71 exact baseline blobs; 9 exact overlays; prior lepton/baryon modules absent')
