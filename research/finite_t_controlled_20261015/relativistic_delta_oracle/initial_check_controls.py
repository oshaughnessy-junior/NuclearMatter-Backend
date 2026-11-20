from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
from oracle import *
checks=[]
def rejects(name,func):
 try:func();ok=False
 except (ValueError,ArithmeticError):ok=True
 checks.append(dict(name=name,passed=ok))
p=POINTS[0]
for field,val in [('T',0),('T',float('nan')),('mua',float('inf')),('Enu',0),('El',.1),('cosine',1),('ml',-1),('anti',1),('Mweak',-1)]:
 d=vars(p).copy();d[field]=val;rejects(f'point rejects {field}={val}',lambda d=d:Point(**d))
for kw in [dict(order=True),dict(order=3),dict(order=4.5),dict(nphi=0),dict(nphi=float('nan')),dict(tail=float('nan')),dict(tail=0),dict(tail=121),dict(current_model='wrong')]:rejects(f'integration rejects {kw}',lambda kw=kw:integrate(p,**kw))
# Timelike q is not silently fed to the spacelike root formula.
d=vars(p).copy();d.update(Enu=2.,El=1.1,ml=1.,cosine=.99);rejects('timelike unsupported',lambda:integrate(Point(**d)))
# Original longitudinal root independently solved by a generic bracketed solver.
from scipy.optimize import brentq
for i,p in enumerate(POINTS):
 _,q,w,*_=setup(p)
 for u in [0.,3.,17.]:
  pa,pb,jac=shell(p,u,.71);r2=2*p.ma*u
  def delta(z):return math.sqrt(p.ma*p.ma+r2+z*z)+w-math.sqrt(p.mb*p.mb+r2+(z+q)**2)
  root=brentq(delta,-1e7,1e7,xtol=1e-10)
  error=abs(root-pa[3])/max(1.,abs(root));checks.append(dict(name=f'point{i} u{u} independent bracketed pz',error=error,tolerance=1e-9,passed=error<1e-9))
Path(__file__).with_name('control_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
