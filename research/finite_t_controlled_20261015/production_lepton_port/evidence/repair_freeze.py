from pathlib import Path
import ast,json,hashlib,datetime,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
pre=json.loads((P/'PRE_QUADRATURE_REPAIR_PIN.json').read_text())
review=json.loads((P/'INITIAL_INDEPENDENT_REVIEW_PIN.json').read_text())
for rel,h in pre.items(): assert sha(P/'pre_quadrature_repair'/rel)==h,rel
for rel,h in review.items(): assert sha(P.parent/'review_production_lepton_port'/rel)==h,rel
paths=sorted(str(p.relative_to(P/'staged')) for p in (P/'staged').rglob('*') if p.is_file() and '__pycache__' not in str(p))
assert len(paths)==5,paths
for rel in paths:
 if rel.endswith('.py'): ast.parse((P/'staged'/rel).read_text(),feature_version=(3,9))
module='src/nmbackend/reference/finite_t_leptons.py'
assert (P/'test_mirror'/module).read_bytes()==(P/'staged'/module).read_bytes()
mirror=json.loads((P/'evidence/mirror_verification.json').read_text())
assert mirror['passed']==71
for c in mirror['checks']: assert c['passed']
for name,expected in [('repair_frozen_tests.xml',81),('repair_frozen_cold.xml',13)]:
 r=ET.parse(P/'evidence'/name).getroot(); suites=[r] if r.tag=='testsuite' else list(r)
 assert sum(int(s.get('tests','0')) for s in suites)==expected
 assert all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
momentum=json.loads((P/'evidence/repair_momentum_results.json').read_text()); assert momentum['passed']==213 and momentum['failed']==0
for name in ['repair_frozen_tests.stderr','repair_frozen_cold.stderr','repair_frozen_momentum.stderr','repair_example.stderr']: assert (P/'evidence'/name).read_bytes()==b''
json.loads((P/'evidence/repair_example.json').read_text())
preservation=dict(initial_snapshot_files_unchanged=len(pre),initial_independent_review_files_unchanged=len(review),
                  initial_review_sha256=sha(P.parent/'review_production_lepton_port/REVIEW.md'),
                  exact_baseline_package_blobs=71,staged_file_count=5,python39_grammar_files=3,
                  candidate_source_sha256=sha(P/'staged'/module),mirror_overlay_matches=True,
                  candidate_tests=81,unchanged_cold_tests=13,momentum_root_checks=213,
                  max_momentum_relative_error=max(x.get('relative_error')or 0 for x in momentum['rows']),
                  max_oracle_refinement=max(x.get('oracle_refinement',0) for x in momentum['rows']))
(P/'evidence/repair_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
manifest=dict(status='AUTHOR_REPAIR_FROZEN_DIFFERENT_INDEPENDENT_RECHECK_REQUIRED',base_commit='ffb5cc72949f2e5401251847aef4c9c7a98dde78',files={rel:sha(P/'staged'/rel) for rel in paths},
              initial_candidate_manifest_sha256=sha(P/'pre_quadrature_repair/CANDIDATE_MANIFEST.json'),
              initial_independent_review_sha256=preservation['initial_review_sha256'],remote_branch_created=False,production_modified=False)
(P/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
allfiles={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and p!=P/'CHECKPOINT.json' and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)}
checkpoint=dict(status=manifest['status'],time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=manifest['base_commit'],candidate_files=manifest['files'],
                tests=dict(candidate_passed=81,unchanged_cold_units_passed=13,momentum_root_passed=213,final_failures=0),
                preserved_initial_four_forward_failures=True,preservation=preservation,production_changes=False,remote_branch_or_PR=False,dependencies_added=False,
                open_gates=['different independent repair recheck','normal admitted full repository CI','minimum/supported version execution and packaging qualification'],files=allfiles)
(P/'CHECKPOINT.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps(dict(preservation=preservation,manifest_sha256=sha(P/'CANDIDATE_MANIFEST.json'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),report_sha256=sha(P/'AUTHOR_REPORT.md')),indent=2))
