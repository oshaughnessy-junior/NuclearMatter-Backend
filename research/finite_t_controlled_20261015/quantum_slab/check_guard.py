"""Verify leaf-local panel guard against unchanged mapped capture where defined."""
from pathlib import Path
import json
import numpy as np
from fixed_bath import *
import guarded_capture
checks=[]
for nn,np_,fl,E in [(7e-5,3e-5,'nue',10.),(7e-5,3e-5,'antinue',.85),(.002,.001,'nue',60.)]:
 m=matter_state(2.,nn,np_);opt=m.optical(fl)
 a=mapped_capture.log_coefficients(opt,E,energy_order=48,angle_order=48)
 b,c,discard=guarded_capture.log_coefficients(opt,E,energy_order=48,angle_order=48)
 for i,v in enumerate((b,c)):
  err=abs(float(np.expm1(v-a[i])))
  checks.append(dict(name=f'{nn},{np_},{fl},{E} coefficient{i} relative',error=err,passed=bool(err<2e-12)))
m=matter_state(2.,.002,.001)
a,b,discard=guarded_capture.log_coefficients(m.optical('nue'),60.,energy_order=72,angle_order=72,tail=50.)
checks.append(dict(name='original failure has recorded collapsed panels',error=float(len(discard)==0),passed=bool(discard)))
checks.append(dict(name='discarded raw widths are unrepresentable scale',error=max(d['width'] for d in discard),passed=all(0<=d['width']<=2e-14 for d in discard)))
r=dict(passed=all(c['passed'] for c in checks),checks=checks,discarded_panels=discard)
Path(__file__).with_name('guard_validation.json').write_text(json.dumps(r,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} guard checks pass")
raise SystemExit(0 if r['passed'] else 1)
