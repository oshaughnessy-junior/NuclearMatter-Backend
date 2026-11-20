from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((P/'current_results.json').read_text());b=json.loads((P/'thermal_results.json').read_text())
assert a['passed']==184 and b['passed']==145 and not a['failed']and not b['failed']
assert b['before']==b['after']
for rel,h in b['before'].items():assert sha(R/rel)==h,rel
for n in ['current.stderr','thermal.stderr']+[f'thermal_{v}_{p}.stderr'for v in ['full','F2zero']for p in [0,1]]:assert (P/n).read_bytes()==b''
files={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file()and p.name!='CHECKPOINT.json'and '__pycache__'not in str(p)}
out=dict(status='ACCEPTED_SELECTED_ONE_BATH_F2_ONLY_RESEARCH_SLICE',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=329,failed_physical_gates=0,retained_initial_review_syntax_error=True,author_source_sha256=sha(R/'current_correction_assessment/configured_currents.py'),author_manifest_sha256=sha(R/'current_correction_assessment/MANIFEST.json'),frozen_author_upstream_fit_paths_unchanged=len(b['before']),review_sha256=sha(P/'REVIEW.md'),files=files)
(P/'CHECKPOINT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(review_sha256=sha(P/'REVIEW.md'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),passed=329,unchanged_paths=len(b['before'])),indent=2))
