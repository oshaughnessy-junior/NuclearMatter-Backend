from pathlib import Path
import ast,json,hashlib,datetime,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
roots={'PRE_QUADRATURE_REPAIR_PIN.json':P/'pre_quadrature_repair','PRE_SCIPY17_REPAIR_PIN.json':P/'pre_scipy17_repair','PRE_HOLE_BOUND_REPAIR_PIN.json':P/'pre_hole_bound_repair','INITIAL_INDEPENDENT_REVIEW_PIN.json':P.parent/'review_production_lepton_port','INDEPENDENT_QUADRATURE_REVIEW_PIN.json':P.parent/'review_production_lepton_repair','INDEPENDENT_COMPATIBILITY_REVIEW_PIN.json':P.parent/'review_production_lepton_compatibility','PRE_UNIT_HOLE_REPAIR_PIN.json':P/'pre_unit_hole_repair','INDEPENDENT_THERMAL_BOUND_REVIEW_PIN.json':P.parent/'review_production_lepton_thermal_bound'}
counts={}
for pin,root in roots.items():
 rows=json.loads((P/pin).read_text());counts[pin]=len(rows)
 for rel,h in rows.items():assert sha(root/rel)==h,(pin,rel)
paths=sorted(str(f.relative_to(P/'staged'))for f in (P/'staged').rglob('*')if f.is_file()and '__pycache__'not in str(f));assert len(paths)==5
for rel in paths:
 if rel.endswith('.py'):ast.parse((P/'staged'/rel).read_text(),feature_version=(3,9))
module='src/nmbackend/reference/finite_t_leptons.py';assert (P/'staged'/module).read_bytes()==(P/'test_mirror'/module).read_bytes()
r=ET.parse(P/'evidence/unit_hole_tests.xml').getroot();suites=[r]if r.tag=='testsuite'else list(r)
assert sum(int(s.get('tests',0))for s in suites)==103
assert all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
probe=json.loads((P/'evidence/unit_hole_results.json').read_text());assert probe['passed']==42 and probe['failed']==0 and probe['source_sha256']==sha(P/'staged'/module)
assert json.loads((P/'evidence/mirror_verification.json').read_text())['passed']==71
for f in ['unit_hole_tests.stderr','unit_hole.stderr']:assert (P/'evidence'/f).read_bytes()==b''
preservation=dict(pin_sets_unchanged=counts,exact_baseline_blobs=71,candidate_source_sha256=sha(P/'staged'/module),candidate_tests=90,cold_tests=13,preservation_functional_checks=42,previous_four_muon_functional_failures_reproduced_and_now_resolved=True)
(P/'evidence/unit_hole_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
manifest=dict(status='AUTHOR_UNIT_HOLE_REPAIR_FROZEN_DIFFERENT_RECHECK_REQUIRED',base_commit='ffb5cc72949f2e5401251847aef4c9c7a98dde78',files={r:sha(P/'staged'/r)for r in paths},prior_manifest_sha256=sha(P/'pre_unit_hole_repair/author/CANDIDATE_MANIFEST.json'),prior_thermal_bound_review_sha256=sha(P.parent/'review_production_lepton_thermal_bound/REVIEW.md'),prior_compatibility_review_sha256=sha(P.parent/'review_production_lepton_compatibility/REVIEW.md'),remote_branch_created=False,production_modified=False)
(P/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={str(f.relative_to(P)):sha(f)for f in P.rglob('*')if f.is_file()and f!=P/'CHECKPOINT.json'and '__pycache__'not in str(f)and '.pytest_cache'not in str(f)}
checkpoint=dict(status=manifest['status'],time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=manifest['base_commit'],candidate_files=manifest['files'],preservation=preservation,production_changes=False,remote_branch_or_PR=False,dependencies_added=False,open_gates=['different independent thermal-bound/functional review','normal admitted repository CI, minimum/supported-version, packaging execution'],files=files)
(P/'CHECKPOINT.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps(dict(source_sha256=sha(P/'staged'/module),manifest_sha256=sha(P/'CANDIDATE_MANIFEST.json'),report_sha256=sha(P/'AUTHOR_REPORT.md'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),preservation=preservation),indent=2))
