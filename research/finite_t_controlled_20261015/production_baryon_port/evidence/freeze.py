from pathlib import Path
import json,hashlib,datetime,ast,difflib
P=Path(__file__).resolve().parents[1];R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((P/'evidence/SOURCE_PIN.json').read_text())
for rel,h in pins.items():assert sha(R/rel)==h,rel
verify=json.loads((P/'evidence/verification.json').read_text());assert verify['forward_checks']==48 and verify['candidate_tests']==36 and verify['cold_tests']==13
module='src/nmbackend/reference/finite_t_baryons.py';assert sha(P/'staged'/module)==verify['candidate_source_sha256']
assert (P/'test_mirror'/module).read_bytes()==(P/'staged'/module).read_bytes()
for item in json.loads((P/'evidence/baseline_package_blobs.json').read_text())['checks']:
 for base in [P/'test_mirror',R/'production_lepton_port/test_mirror']:
  b=(base/item['path']).read_bytes();assert hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()==item['expected_git_blob']
paths=sorted(str(p.relative_to(P/'staged'))for p in (P/'staged').rglob('*')if p.is_file()and '__pycache__'not in str(p));assert len(paths)==5
for rel in paths:
 if rel.endswith('.py'):ast.parse((P/'staged'/rel).read_text(),feature_version=(3,9))
(P/'evidence/README.diff').write_text(''.join(difflib.unified_diff((P/'baseline/README.md').read_text().splitlines(True),(P/'staged/README.md').read_text().splitlines(True),fromfile='a/README.md',tofile='b/README.md')))
manifest=dict(status='AUTHOR_FROZEN_DIFFERENT_INDEPENDENT_REVIEW_REQUIRED',base_commit='ffb5cc72949f2e5401251847aef4c9c7a98dde78',files={rel:sha(P/'staged'/rel)for rel in paths},production_modified=False,remote_branch_created=False,dependencies_added=False)
(P/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file()and p!=P/'CHECKPOINT.json'and '__pycache__'not in str(p)and '.pytest_cache'not in str(p)}
checkpoint=dict(status=manifest['status'],time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=manifest['base_commit'],candidate_files=manifest['files'],tests=dict(forward_checks=48,candidate_tests=36,unchanged_cold_tests=13,numerical_failures=0),exact_baseline_package_blobs=71,real_package_module_origins=24,sourcepins_unchanged=pins,existing_lepton_mirror_blobs_verified_unchanged=71,production_modified=False,remote_branch_or_PR=False,dependencies_added=False,open_gates=['different independent physics/numerical/API review','normal repository CI/full suite','supported/minimum-version and pin-environment execution','packaging'],files=files)
(P/'CHECKPOINT.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps(dict(source_sha256=sha(P/'staged'/module),manifest_sha256=sha(P/'CANDIDATE_MANIFEST.json'),report_sha256=sha(P/'AUTHOR_REPORT.md'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),staged_paths=paths),indent=2))
