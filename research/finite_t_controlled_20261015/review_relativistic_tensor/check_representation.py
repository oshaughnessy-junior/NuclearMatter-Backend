"""Actual tiny coefficient scaling/flags, outside ordinary oracle tail scope."""
from pathlib import Path
import sys,math,json,hashlib,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'relativistic_tensor'));import tensor as t
checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
ma,mb=939.56542052,938.27208816;T=.05;mu_a=ma-30*T;mu_b=mb-1500*T
for shift in (0.,-1000*T):
 state=t.State(T,ma,mb,mu_a+shift,mu_b);r=t.differential(state,47.,46.,.13,.51099895069,20.,energy_order=40,phi_order=16,tail=80.);row=dict(shift=shift,j=r['j'],chi=r['chi'],abs_scale=r['response']['absorption']['log_scale'],em_scale=r['response']['emission']['log_scale'],abs_scaled=r['response']['absorption']['scaled'],em_scaled=r['response']['emission']['scaled'],abs_reason=r['response']['absorption']['reason']);rows.append(row)
 gate('actual finite log chi '+str(shift),float(not math.isfinite(r['chi']['log'])),.1);gate('actual finite log j '+str(shift),float(not math.isfinite(r['j']['log'])),.1)
 gate('actual bothoccupations log KMS '+str(shift),abs(r['j']['log']-r['chi']['log']+(47.-(20.+mu_b-(mu_a+shift)))/T),5e-11)
 if shift:
  gate('actual absorption underflow flag',float(r['chi']['value']is not None or r['chi']['representable']or r['chi']['reason']!='underflow'),.1)
  gate('actual tensor component underflow flag',float(r['response']['absorption']['tensor']is not None or r['response']['absorption']['representable']),.1)
# In the selected dilute initial state, fa<<1 while final occupation remains fixed.
gate('dilute fixedsupport log scaling',abs(rows[1]['chi']['log']-rows[0]['chi']['log']+1000.),5e-10)
gate('dilute fixedsupport scaled allW',np.max(abs(rows[1]['abs_scaled']-rows[0]['abs_scaled']))/np.max(abs(rows[0]['abs_scaled'])),3e-11)
# Exact structural-empty support has distinct metadata from an underflow.
s=t.State(4.,939.,939.,940.,940.);r=t.response(s,60.,50.);gate('empty support separately labeled',float(r['support']!=[]or r['absorption']['reason']!='empty_support'or r['absorption']['log_scale']is not None),.1)
def serialize(x):
 if isinstance(x,np.ndarray):return dict(real=x.real.tolist(),imag=x.imag.tolist())
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
(P/'representation_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,scope='Representation and analytic dilute scaling witnesses; no independent extreme-tail relative accuracy certificate'),indent=2,default=serialize)+'\n');print(sum(c['passed']for c in checks),'/',len(checks))
for c in checks:
 if not c['passed']:print('FAIL',c)
