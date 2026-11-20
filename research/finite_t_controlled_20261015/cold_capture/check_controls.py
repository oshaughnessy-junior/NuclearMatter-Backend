from pathlib import Path
import json,math
import resolved as r
s=r.model.make_state(r.model.cold_reference(.01,.1),.01).optical('nue');checks=[]
def reject(name,call,kind=ValueError):
 try:call()
 except kind:checks.append(dict(name=name,passed=True))
 else:checks.append(dict(name=name,passed=False))
for fn in [r.adaptive_yq,r.gauss_Ec]:
 for kw in [dict(Enu=math.nan),dict(Enu=math.inf),dict(Enu=0.),dict(Enu=-1),dict(order=True),dict(order=12.5),dict(order=4),dict(tail=math.nan),dict(tail=59.),dict(tail=241.)]:
  reject(fn.__name__+str(kw),lambda fn=fn,kw=kw:fn(s,**({'Enu':.03}|kw)))
 lowmu=r.A.OpticalState(s.baryons,0.)
 reject(fn.__name__+' lowmu domain',lambda fn=fn:fn(lowmu,.03))
reject('rtol invalid',lambda:r.adaptive_yq(s,.03,rtol=math.nan))
reject('collapsed q coordinate',lambda:r.adaptive_yq(s,1e-30),ArithmeticError)
for log in [math.nan,math.inf,-math.inf]:reject('nonfinite result'+str(log),lambda log=log:r.represent(log),ArithmeticError)
for log,expect in [(-1000.,False),(-745.,True),(710.,False)]:checks.append(dict(name='representability'+str(log),passed=r.represent(log)['representable']==expect))
rpt=dict(passed=all(c['passed']for c in checks),checks=checks)
Path(__file__).with_name('controls.json').write_text(json.dumps(rpt,indent=2)+'\n');print(len(checks),rpt['passed'])
