"""Verify frozen prerequisite union and emit additive eight-path candidate diff."""
from pathlib import Path
import hashlib,json,difflib
P=Path(__file__).resolve().parent;S=P/'staged';R=P.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((P/'STAGED_BASE_PIN.json').read_text())
new=['src/nmbackend/reference/free_matter.py','src/nmbackend/reference/_free_matter_sources.py','tests/test_free_matter_reference.py','tests/test_free_matter_sources.py','tests/data/free_matter_reference_internal.json','examples/free_matter_reference.py','docs/free_matter_reference.md']
changed=[f for f,h in base.items() if sha(S/f)!=h];assert changed==['README.md'],changed
baseline=R/'nmb-finite-t-20261015/production_weak_packaging_review/package_copy/README.md'
assert sha(baseline)==base['README.md']
text=''.join(difflib.unified_diff(baseline.read_text().splitlines(True),(S/'README.md').read_text().splitlines(True),fromfile='a/README.md',tofile='b/README.md'))
for f in new:
 assert f not in base and (S/f).exists()
 text+=''.join(difflib.unified_diff([],(S/f).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+f))
(P/'PROPOSED_DIFF.patch').write_text(text)
(P/'CHANGED_PATHS.json').write_text(json.dumps(dict(base='Explicit prerequisite union in PREREQUISITES.json; not main',changed=['README.md']+new,baseline_files_checked=len(base),unchanged_baseline_files=len(base)-1,hashes={f:sha(S/f) for f in ['README.md']+new}),indent=2)+'\n')
print(json.dumps(dict(candidate_paths=8,unchanged_baseline_files=len(base)-1,diff_sha256=sha(P/'PROPOSED_DIFF.patch'))))
