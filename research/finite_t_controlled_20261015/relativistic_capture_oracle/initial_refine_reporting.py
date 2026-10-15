from pathlib import Path
import sys,json,math,time
sys.dont_write_bytecode=True
from events import *
HERE=Path(__file__).resolve().parent;start=time.monotonic();d=json.loads((HERE/'results.json').read_text());old=[r for r in d['rows']if r['label']=='warm'][-1]['result']
def checkpoint(v):
 (HERE/'refinement_partial.json').write_text(json.dumps(dict(partial=v,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
 if time.monotonic()-start>100:raise TimeoutError('Bounded warm refinement')
v=integrate(WARM,20,20,32,16,80,checkpoint=checkpoint);checks=[]
for name in ['chi','j']:
 err=math.expm1(old[name]['log']-v[name]['log']);checks.append(dict(name=name+'14→20 warm refinement',error=err,tolerance=3e-6,passed=abs(err)<3e-6))
err=v['density']/6.999999999999623e-5-1;checks.append(dict(name='fine warm forward density',error=err,tolerance=3e-9,passed=abs(err)<3e-9))
err=v['j']['log']-v['chi']['log']+(WARM.Enu-(WARM.mul+WARM.mub-WARM.mua))/WARM.T;checks.append(dict(name='fine separately integrated inverse KMS',error=err,tolerance=3e-10,passed=abs(err)<3e-10))
(HERE/'refinement.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,result=v,runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(checks,time.monotonic()-start)
