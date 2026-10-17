"""Exact old/repaired raw physical replay comparison, ignoring wall-clock timings."""
from pathlib import Path
import json,hashlib
P=Path(__file__).parent;old=P/'initial_input_failure_snapshot'
def strip(x):
 if isinstance(x,dict):return{k:strip(v)for k,v in x.items()if k!='seconds'}
 if isinstance(x,list):return[strip(v)for v in x]
 return x
checks=[]
for name in ['results.json','accepted_results.json','partial_grids.json','refinement_partial.json']:
 before=json.loads((old/name).read_text());after=json.loads((P/name).read_text());checks.append(dict(name='exact physical raw replay '+name,passed=strip(before)==strip(after)))
pins=json.loads((old/'SNAPSHOT_PINS.json').read_text());checks.append(dict(name='original snapshot exactallfiles',passed=all(hashlib.sha256((old/n).read_bytes()).hexdigest()==sha for n,sha in pins.items())))
raw=json.loads((P/'accepted_results.json').read_text());checks.append(dict(name='frozen upstreamphysical sourcebytes exact',passed=all(hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==sha for n,sha in raw['pins'].items())))
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']]);(P/'replay_audit_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
