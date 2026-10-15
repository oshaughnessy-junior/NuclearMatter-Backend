"""Reproduce reviewer counterexamples and unchanged finite-input algebra."""
from __future__ import annotations
import importlib.util,hashlib,json,sys
from pathlib import Path
import numpy as np
import external_legs as w
HERE=Path(__file__).resolve().parent
oldpath=HERE/'pre_helper_repair'/'external_legs.py'
# The preserved module resolves its strong dependency one level differently;
# executing its exact bytes under original __file__ keeps that dependency fixed.
old=type(sys)('ope_weak_original_helper_replay')
old.__file__=str(HERE/'external_legs.py')
exec(compile(oldpath.read_bytes(),str(oldpath),'exec'),old.__dict__)
checks=[]
def gate(name,passed,**evidence):checks.append(dict(name=name,passed=bool(passed),**evidence))
p=w.emission_point(117.,6.1,[2.,-3.,1.],[.23,.51,-.37]);Q=np.array([2.,-3.,1.]);om=6.1
oldnan=old.vector_defect_direct(p,om,Q,gv=float('nan'))
oldraising=old.vector_defect_direct(p,om,Q,raising=float('nan'))
gate('original gvNaN counterexample reproduced',bool(np.any(np.isnan(oldnan))))
gate('original raisingNaN counterexample reproduced',bool(np.all(np.isfinite(oldraising))))
for name,value in [('nan',float('nan')),('posinf',float('inf')),('neginf',-float('inf')),('complex',1+0j),('bool',True),('string','1')]:
 for api,call in [('defect',lambda:w.vector_defect_direct(p,om,Q,gv=value)),('mainV',lambda:w.amplitude(p,om,Q,'V0',gv=value)),('mainA',lambda:w.amplitude(p,om,Q,'Ax',ga=value))]:
  try:call();message='NOT REJECTED'
  except ValueError as e:message=str(e)
  gate(api+' rejects '+name,message!='NOT REJECTED',message=message)
for value in [float('nan'),float('inf'),0,1,'False',None,1+0j]:
 for api,call in [('rho',lambda:w.rho(0,raising=value)),('defect',lambda:w.vector_defect_direct(p,om,Q,raising=value)),('main',lambda:w.amplitude(p,om,Q,'V0',raising=value))]:
  try:call();message='NOT REJECTED'
  except ValueError as e:message=str(e)
  gate(api+' rejects nonBoolean '+repr(value),message!='NOT REJECTED',message=message)
for raising in [False,True,np.bool_(False),np.bool_(True)]:
 for gv in [-1.3,0.,np.float64(.81),1.]:
  a=w.vector_defect_direct(p,om,Q,gv=gv,raising=raising)
  b=old.vector_defect_direct(p,om,Q,gv=gv,raising=raising)
  gate('finite defect bitwise unchanged '+str((bool(raising),float(gv))),np.array_equal(a,b))
  for comp in ['V0','Vx','Ax']:
   a=w.amplitude(p,om,Q,comp,gv=gv,raising=raising);b=old.amplitude(p,om,Q,comp,gv=gv,raising=raising)
   gate('finite amplitude bitwise unchanged '+str((bool(raising),float(gv),comp)),np.array_equal(a,b))
result=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],old_source_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),new_source_sha256=hashlib.sha256((HERE/'external_legs.py').read_bytes()).hexdigest(),scope='helper/current-input repair only; finite-input equations unchanged')
(HERE/'helper_repair_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k]for k in ['passed','failed','old_source_sha256','new_source_sha256']}))
if result['failed']:raise SystemExit(1)
