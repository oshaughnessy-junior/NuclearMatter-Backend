"""Reversal, dimensions and explicit defect-helper input limits."""
from pathlib import Path
import json,math
import numpy as np
from probe import W,point,checks,gate,compare,HERE,snapshot,before
start=len(checks);rows=[];p,Q=point(117.3,6.7,[-1.7,2.9,.8],[.24,-.61,.75]);om=6.7
for comp in['V0','Vx','Vy','Vz','Ax','Ay','Az']:
 f=W.amplitude(p,om,Q,comp);r=W.amplitude(p[[2,3,0,1]],-om,-Q,comp,raising=True);compare('fresh reciprocal charged adjoint '+comp,r,f.conj().T)
 lam=2.17;model=W.ope.Model(1.01,140*lam,939*lam);a=W.amplitude(p*lam,om*lam,Q*lam,comp,model=model,pole_gap=.05*lam);compare('fresh MeV^-3 dimensional scaling '+comp,a*lam**3,f)
# Helpers should not silently reinterpret non-Boolean weak charge or returnNaN.
controls=[]
for name,kwargs in [('nan_gv',dict(gv=float('nan'))),('nan_raising',dict(raising=float('nan')))]:
 try:
  a=W.vector_defect_direct(p,om,Q,**kwargs);rejected=False;finite=bool(np.all(np.isfinite(a)))
 except ValueError:rejected=True;finite=None
 controls.append(dict(name=name,rejected=rejected,finite=finite))
 # Record actual failure without changing author code or broadening physics scope.
 gate('defect helper rejects '+name,float(not rejected),0)
gate('source unchanged including original complex failure',int(snapshot()!=before),0)
out=dict(checks=checks[start:],passed=sum(c['passed']for c in checks[start:]),failed=[c for c in checks[start:]if not c['passed']],defect_helper_controls=controls,author_after=snapshot())
(HERE/'extra_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','defect_helper_controls']},indent=2))
