"""Unintegrated exact-coordinate Jacobian gate; not a spectral profile call."""
from pathlib import Path
import json,math,numpy as np
from conditional_beta import ENDPOINT,BATH
from beta_pr55_reference import _weak_current as C
rows=[]
for E in (1.2,1.3):
 eta0=math.log(E/ENDPOINT);en0=C.MN*math.cosh(eta0);p0=C.MN*math.sinh(eta0);T=BATH[0]
 def coordinates(z,v):
  en=en0+T*z;p=math.sqrt((en-C.MN)*(en+C.MN));R=T*z*(1+(en+en0)/(p+p0));width=R/p
  return np.array([p,1-width*(1-v*v)]),2*T*en*width*v/p
 z,v=2.3,.61;central,pred=coordinates(z,v)
 for h in (.01,.003,.001,.0003):
  jac=np.column_stack([(coordinates(z+h,v)[0]-coordinates(z-h,v)[0])/(2*h),(coordinates(z,v+h)[0]-coordinates(z,v-h)[0])/(2*h)])
  val=abs(float(np.linalg.det(jac)));rows.append({'E':E,'z':z,'v':v,'h':h,'analytic_dp_dc_over_dz_dv':pred,'numeric_jacobian':val,'relative_error':val/pred-1})
  print(E,h,val/pred-1)
Path(__file__).with_name('PHASE_JACOBIAN_RAW.json').write_text(json.dumps(rows,indent=2)+'\n')
