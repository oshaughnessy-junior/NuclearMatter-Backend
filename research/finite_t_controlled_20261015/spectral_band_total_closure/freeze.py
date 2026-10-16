"""Saved-only canonical evidence and source-preserving freeze; no package imports."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda n:json.loads((R/n).read_text())
write=lambda n,a:(R/n).write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
pins=load('SOURCE_PIN.json')
for name,digest in pins.items():
    assert sha(R.parent/name)==digest,name
assert sha(R/'LOCK.json')=='cc7a622aa0cd4ea4afdffbbf719a146ff0cb807c49fd61b6b462cb2913685d90'
a=load('ANALYSIS.json')
counts={}
for ch in ('EC','PC','NEUTRON_DECAY'):
    jobs=[load(str(p.relative_to(R))) for p in sorted((R/'raw').glob(ch+'_*.json'))]
    counts[ch]={'jobs':len(jobs),'complete_jobs':sum(j['complete'] for j in jobs),'complete_coefficient_nodes':sum(len(j.get('rows',[])) for j in jobs),'incomplete_jobs':sum(not j['complete'] for j in jobs)}
assert counts['EC']['complete_coefficient_nodes']==160
assert counts['PC']['complete_coefficient_nodes']==0
assert counts['NEUTRON_DECAY']['complete_coefficient_nodes']==160
bands={ch:load(ch+'_BAND_RESULTS.json') for ch in counts}
assert all(not x['complete_declared_band'] and x['band_moments'] is None for x in bands['PC'])
canonical=dict(kind='SAVED actual finite-band diagnostic, not complete-source API or runtime cache',protocol=load('LOCK.json'),source_pin_sha256=sha(R/'SOURCE_PIN.json'),counts=counts,bands=bands,fresh_event_totals=a['fresh_event_totals'],channel_status=a['channel_status'],failed=a['failed'],blocked=a['blocked'],whole_moment_qualified=False,independent_review='REQUIRED, pending at author freeze')
write('CANONICAL_BAND_DIAGNOSTIC.json',canonical)
entries={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','CHECKPOINT.json') and '__pycache__' not in p.parts}
write('MANIFEST.json',dict(files=entries,excluded=['MANIFEST.json self','CHECKPOINT.json final identity','__pycache__ generated interpreter cache only'],source_pins_verified_unchanged=True))
write('CHECKPOINT.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FROZEN; selected EC numerical agreement, PC incomplete and beta control failures; DIFFERENT review pending',manifest_sha256=sha(R/'MANIFEST.json'),report_sha256=sha(R/'REPORT.md'),analysis_sha256=sha(R/'ANALYSIS.json'),canonical_sha256=sha(R/'CANONICAL_BAND_DIAGNOSTIC.json'),lock_sha256=sha(R/'LOCK.json'),source_pin_sha256=sha(R/'SOURCE_PIN.json'),passed=a['passed'],numeric_failed=len(a['failed']),blocked=len(a['blocked']),counts=counts,source_pins_verified_unchanged=True,whole_moment_qualified=False,no_further_scientific_calls_planned=True))
print(json.dumps(load('CHECKPOINT.json'),indent=2))
