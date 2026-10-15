from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
from events import *
checks=[]
def rejects(n,f):
 try:f();ok=False
 except (ValueError,ArithmeticError):ok=True
 checks.append(dict(name=n,passed=ok))
for key,value in [('T',0),('T',float('nan')),('mua',float('inf')),('Enu',0),('ml',-1),('anti',1),('Mweak',0)]:
 d=vars(WARM).copy();d[key]=value;rejects(str((key,value)),lambda d=d:Bath(**d))
for kw in [dict(radial=True),dict(incident=3),dict(outgoing=4.5),dict(azimuth=0),dict(tail=float('inf')),dict(tail=19),dict(tail=101)]:rejects(str(kw),lambda kw=kw:integrate(WARM,**kw))
for v in [float('nan'),float('inf'),-float('inf')]:rejects('nonfinite returned log '+str(v),lambda v=v:represented(v))
for x in [-800.,800.]:
 r=represented(x);checks.append(dict(name='finite unrepresented log '+str(x),passed=not r['representable']and r['value']is None and r['log']==x))
Path(__file__).with_name('control_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
