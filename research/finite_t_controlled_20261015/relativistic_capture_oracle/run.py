from pathlib import Path
import sys,json,time,math,hashlib,atexit
sys.dont_write_bytecode=True
from events import *
HERE=Path(__file__).resolve().parent;start=time.monotonic();rows=[];checks=[];phase='start';partial=None
def save():
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,partial=partial,rows=rows,checks=checks,runtime_seconds=time.monotonic()-start,source_sha256=hashlib.sha256((HERE/'events.py').read_bytes()).hexdigest()),indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def checkpoint(v):
 global partial
 partial=v;save()
 if time.monotonic()-start>105:raise TimeoutError('Bounded event experiment')
for label,b,density in [('dilute',DILUTE,7.000000000001131e-16),('warm',WARM,6.999999999999623e-5)]:
 vals=[]
 for order in [(6,6,10,6,30),(10,10,16,8,45),(14,14,24,12,60)]:
  phase=label+str(order);v=integrate(b,*order,checkpoint=checkpoint);vals.append(v);rows.append(dict(label=label,bath=vars(b),result=v));save()
 for i,v in enumerate(vals[:-1]):
  for name in ['chi','j']:gate(label+f' {i} {name} refinement',math.expm1(v[name]['log']-vals[-1][name]['log']),3e-6)
 v=vals[-1];gate(label+' forward density',v['density']/density-1,3e-9);gate(label+' four momentum',v['max_fourmomentum_residual'],2e-14)
 expected=-(b.Enu-(b.mul+b.mub-b.mua))/b.T;gate(label+' separately integrated inverse KMS',v['j']['log']-v['chi']['log']-expected,3e-10)
phase='complete';partial=None;save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
