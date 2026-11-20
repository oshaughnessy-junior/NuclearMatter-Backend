"""Lock independent review before candidate comparisons or event source calls."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;A=P.parent/'installed_weak_density_element';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(A/'LOCK.json')=='55b82078ca4aeb2195ebc000704904694e543f0c50cdefd6ff6f351b3d3f0391'
lock=json.loads((A/'LOCK.json').read_text())
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_imported=False,source_calls_made=False,author_lock_sha=sha(A/'LOCK.json'),method='RK4 independently assembled equations and accumulators in SAME specificu/Ye variables asauthor midpoint; not an independentphysicalmodel/EOS/amplitude.',steps=[2,4],source_profile='base',initial=lock['initial'],density=lock['density'],gates=lock['gates'],endpoint_source_profile='fine',no_admission_expansion=True,prior_binding_review='11295739f71fe779553a3289bfc445cd44906b551a7ea7809b08968ce6fa25c7',clarification='PREPARATION_LOCK predatesauthorLOCK and incorrectlyanticipatedauthorTdotcoordinates. ActualauthorusesuMidpoint. Preserveinitialplan, correctindependenceclaimherebeforedata.')
(P/'VALIDATION_LOCK.json').write_text(json.dumps(r,indent=2)+'\n')
print('LOCKED review',sha(P/'VALIDATION_LOCK.json'))
