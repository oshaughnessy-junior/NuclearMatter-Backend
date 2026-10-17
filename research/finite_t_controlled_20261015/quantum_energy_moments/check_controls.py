"""Finite/positive public integration controls; no expensive kernels invoked."""
import json,math
from integrated import *
m=slab.matter_state(2.,7e-5,3e-5);checks=[]
cases=[dict(length_cm=-1),dict(length_cm=math.inf),dict(outer_order=3),dict(outer_order=True),dict(internal_order=math.nan),dict(tail=-1),dict(tail=math.nan),dict(edges=[0,1,math.inf]),dict(edges=[0,1,1]),dict(edges=[1,2]),dict(flavor='numu')]
for kw in cases:
 a=dict(flavor='nue');a.update(kw)
 try:integrate(m,**a)
 except ValueError:checks.append(dict(case=str(kw),passed=True))
 else:checks.append(dict(case=str(kw),passed=False))
Path(__file__).with_name('control_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(sum(c['passed'] for c in checks),'/',len(checks));raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
