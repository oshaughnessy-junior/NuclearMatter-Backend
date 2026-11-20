"""Selected finite-current input and MeV dimensional controls; review-only."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('current_control_review',HERE.parent/'ope_vector_current/current.py');c=importlib.util.module_from_spec(s);sys.modules[s.name]=c;s.loader.exec_module(c)
k1=np.array([12.,-21.,17.]);k2=np.array([-9.,4.,23.]);checks=[]
bad=[('gv_nan',lambda:c.direct(k1,k2,gv=float('nan'))),('gv_inf',lambda:c.direct(k1,k2,gv=float('inf'))),('gv_bool',lambda:c.direct(k1,k2,gv=True)),('gv_complex',lambda:c.direct(k1,k2,gv=1+0j)),('raising_nan',lambda:c.direct(k1,k2,raising=float('nan'))),('raising_integer',lambda:c.direct(k1,k2,raising=1)),('transfer_nan',lambda:c.direct([0,0,float('nan')],k2)),('transfer_complex',lambda:c.direct(k1.astype(complex)+1j,k2)),('transfer_shape',lambda:c.direct([1,2],k2)),('antisym_complex',lambda:c.antisymmetrized(np.full((4,3),1j)))]
for name,fn in bad:
 try:fn();msg='NOT REJECTED'
 except ValueError as e:msg=str(e)
 checks.append(dict(name=name,passed=msg!='NOT REJECTED',message=msg))
for raising in [False,True]:
 for lam in [.73,1.37]:
  mod=c.W.ope.Model(1.01,140*lam,939*lam);a=c.direct(k1,k2,raising=raising);b=c.direct(k1*lam,k2*lam,model=mod,raising=raising)
  for kind in ['seagull','pion_flight','total']:
   err=float(np.max(abs(b[kind]*lam**3-a[kind]))/np.max(abs(a[kind])))
   checks.append(dict(name=str((raising,lam,kind))+' independent MeV^-3 scale',error=err,tolerance=3e-12,passed=err<3e-12))
result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']])
(HERE/'control_scale_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','failed']}))
