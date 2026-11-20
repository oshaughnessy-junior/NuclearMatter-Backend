from pathlib import Path
import math,json
import log_gain as r
s=r.nc.state_density(.08,.0042,0.);rad=r.Radiation('FD',.4,5.);checks=[]
def reject(name,fn,kind=ValueError):
 try:fn()
 except kind:checks.append(dict(name=name,passed=True))
 else:checks.append(dict(name=name,passed=False))
for order in [True,7,8.5]:reject('energyorder'+str(order),lambda order=order:r.integrate(s,19.6,rad,energy_order=order))
for order in [True,7,8.5]:reject('angleorder'+str(order),lambda order=order:r.integrate(s,19.6,rad,angle_order=order))
for E in [0.,-1.,math.nan,math.inf]:reject('Ein'+str(E),lambda E=E:r.integrate(s,E,rad))
for Ep in [0.,-1.,math.nan,math.inf]:reject('Eout'+str(Ep),lambda Ep=Ep:r.point(s,19.6,Ep,0.,rad))
for cutoff in [19.6,0.,math.nan]:reject('cutoff'+str(cutoff),lambda cutoff=cutoff:r.integrate(s,19.6,rad,cutoff=cutoff))
for mode in ['bad',None]:reject('reverse'+str(mode),lambda mode=mode:r.integrate(s,19.6,rad,reverse=mode))
for log in [math.nan,math.inf,-math.inf]:reject('nonfiniteordinary'+str(log),lambda log=log:r.represented(log),ArithmeticError)
for pair in [(None,math.nan),(math.inf,None),(0.,math.nan)]:reject('nonfinitedifference'+str(pair),lambda pair=pair:r.difference(*pair),ArithmeticError)
for log,flag in [(-1000.,False),(-745.,True),(710.,False)]:checks.append(dict(name='representability'+str(log),passed=r.represented(log)['representable']==flag))
checks.append(dict(name='algebraiczero',passed=r.represented(None,0)['value']==0 and r.difference(1.,1.)==(0,None)))
checks.append(dict(name='finitehugecancellationlogs',passed=all(math.isfinite(r.difference(a,b)[1])for a,b in [(-3000.,-2280.),(-2280.,-3000.)])))
report=dict(passed=all(c['passed']for c in checks),checks=checks);Path(__file__).with_name('controls.json').write_text(json.dumps(report,indent=2)+'\n');print(len(checks),report['passed'])
