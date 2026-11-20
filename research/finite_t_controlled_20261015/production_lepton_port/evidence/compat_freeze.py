from pathlib import Path
import ast,json,hashlib,datetime,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins=[('PRE_QUADRATURE_REPAIR_PIN.json',P/'pre_quadrature_repair'),('PRE_SCIPY17_REPAIR_PIN.json',P/'pre_scipy17_repair'),('INITIAL_INDEPENDENT_REVIEW_PIN.json',P.parent/'review_production_lepton_port'),('INDEPENDENT_QUADRATURE_REVIEW_PIN.json',P.parent/'review_production_lepton_repair')]
counts={}
for name,root in pins:
 pin=json.loads((P/name).read_text());counts[name]=len(pin)
 for rel,h in pin.items():assert sha(root/rel)==h,(name,rel)
paths=sorted(str(f.relative_to(P/'staged'))for f in (P/'staged').rglob('*')if f.is_file() and '__pycache__' not in str(f));assert len(paths)==5
for rel in paths:
 if rel.endswith('.py'):ast.parse((P/'staged'/rel).read_text(),feature_version=(3,9))
module='src/nmbackend/reference/finite_t_leptons.py';assert (P/'staged'/module).read_bytes()==(P/'test_mirror'/module).read_bytes()
source=ast.parse((P/'staged'/module).read_text())
assert [a.name for n in ast.walk(source)if isinstance(n,ast.ImportFrom)and n.module=='scipy.special'for a in n.names]==['expit']
r=ET.parse(P/'evidence/compat_tests.xml').getroot();suites=[r]if r.tag=='testsuite'else list(r)
assert sum(int(s.get('tests',0))for s in suites)==95
assert all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
checks=json.loads((P/'evidence/compat_import_results.json').read_text());assert checks['passed']==259 and checks['failed']==0
assert checks['source_sha256']==sha(P/'staged'/module)
momentum=json.loads((P/'evidence/compat_momentum_results.json').read_text());assert momentum['passed']==213 and momentum['failed']==0
mirror=json.loads((P/'evidence/mirror_verification.json').read_text());assert mirror['passed']==71
for f in ['compat_tests.stderr','compat_import.stderr','compat_momentum.stderr','compat_thermal_guard_diagnostic.stderr']:assert (P/'evidence'/f).read_bytes()==b''
preservation=dict(pin_sets_unchanged=counts,exact_baseline_blobs=71,candidate_source_sha256=sha(P/'staged'/module),staged_paths=paths,
 candidate_tests=82,cold_tests=13,algebra_import_preservation_checks=259,momentum_checks=213,matched_thermal_rejections_not_forward_passes=2,
 initial_independent_review_sha256=sha(P.parent/'review_production_lepton_port/REVIEW.md'),quadrature_review_sha256=sha(P.parent/'review_production_lepton_repair/REVIEW.md'))
(P/'evidence/compat_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
manifest=dict(status='AUTHOR_COMPATIBILITY_REPAIR_FROZEN_DIFFERENT_RECHECK_REQUIRED',base_commit='ffb5cc72949f2e5401251847aef4c9c7a98dde78',files={r:sha(P/'staged'/r)for r in paths},
 prior_manifest_sha256=sha(P/'pre_scipy17_repair/CANDIDATE_MANIFEST.json'),initial_independent_review_sha256=preservation['initial_independent_review_sha256'],quadrature_review_sha256=preservation['quadrature_review_sha256'],remote_branch_created=False,production_modified=False)
(P/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={str(f.relative_to(P)):sha(f)for f in P.rglob('*')if f.is_file()and f!=P/'CHECKPOINT.json'and '__pycache__'not in str(f)and '.pytest_cache'not in str(f)}
checkpoint=dict(status=manifest['status'],time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=manifest['base_commit'],candidate_files=manifest['files'],preservation=preservation,
 production_changes=False,remote_branch_or_PR=False,dependencies_added=False,
 open_gates=['different independent compatibility/numerical recheck','review public thermal domain/guard at resolved dilute signed m/T600 states','normal admitted repository CI, supported/minimum-version and packaging execution'],files=files)
(P/'CHECKPOINT.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps(dict(source_sha256=sha(P/'staged'/module),manifest_sha256=sha(P/'CANDIDATE_MANIFEST.json'),report_sha256=sha(P/'AUTHOR_REPORT.md'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),preservation=preservation),indent=2))
