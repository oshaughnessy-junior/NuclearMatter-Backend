from pathlib import Path
import json,math
import redistribution as r
s=r.nc.state_density(2.,.002,0.);rad=r.Radiation();checks=[]
def reject(name,fn):
 try:fn()
 except ValueError:checks.append(dict(name=name,passed=True))
 else:checks.append(dict(name=name,passed=False))
for kwargs in [dict(T=0.),dict(T=-1.),dict(T=math.nan),dict(mu=math.inf),dict(kind='unknown')]:reject('radiation '+str(kwargs),lambda kwargs=kwargs:r.Radiation(**kwargs))
for fn in [r.adaptive,r.energy_grid]:
 for E in [0.,-1.,math.nan]:reject(fn.__name__+' Ein'+str(E),lambda fn=fn,E=E:fn(s,E,rad))
 for tail in [29.,101.,math.nan]:reject(fn.__name__+' tail'+str(tail),lambda fn=fn,tail=tail:fn(s,10.,rad,tail=tail))
for order in [True,12.5,4]:reject('angular'+str(order),lambda order=order:r.integrand(s,10.,9.,rad,angle_order=order))
reject('negative Eout',lambda:r.integrand(s,10.,-1.,rad));reject('nonfinite Eout',lambda:r.integrand(s,10.,math.nan,rad));reject('unknown angular coordinate',lambda:r.integrand(s,10.,9.,rad,coordinate='x'))
reject('invalid tolerance',lambda:r.adaptive(s,10.,rad,rtol=math.nan));reject('unknown grid',lambda:r.energy_grid(s,10.,rad,kind='x'))
rpt=dict(passed=all(c['passed']for c in checks),checks=checks);Path(__file__).with_name('controls.json').write_text(json.dumps(rpt,indent=2)+'\n');print(len(checks),rpt['passed'])
