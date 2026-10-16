from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((P/'roots_results.json').read_text());b=json.loads((P/'sources_results.json').read_text())
assert a['passed']==85 and b['passed']==97 and not a['failed']and not b['failed']
assert a['before']==a['after']==b['before']==b['after']
for rel,h in a['before'].items():assert sha(R/rel)==h,rel
for n in ['roots.stderr','sources.stderr']:assert (P/n).read_bytes()==b''
files={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file()and p.name!='CHECKPOINT.json'and '__pycache__'not in str(p)}
out=dict(status='ACCEPTED_SELECTED_TWO_BATH_RESEARCH_ASSESSMENT_ONLY',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks_passed=182,checks_failed=0,frozen_input_files_unchanged=len(a['before']),source_sha256=sha(R/'fit_balance_assessment/assess.py'),author_manifest_sha256=sha(R/'fit_balance_assessment/MANIFEST.json'),review_sha256=sha(P/'REVIEW.md'),files=files)
(P/'CHECKPOINT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(review_sha256=sha(P/'REVIEW.md'),checkpoint_sha256=sha(P/'CHECKPOINT.json'),passed=182,unchanged_inputs=len(a['before'])),indent=2))
