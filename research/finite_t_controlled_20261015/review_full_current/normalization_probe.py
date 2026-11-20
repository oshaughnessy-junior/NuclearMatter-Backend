"""t-coordinate phase space and explicit-spinor crossing/zero-limit checks."""
from pathlib import Path
import sys,json,math,time,runpy
import numpy as np
from scipy.integrate import quad
# Load only definitions from independent review; do not replay its case loop.
P=Path(__file__).resolve().parent;code=(P/'spinor_probe.py').read_text().split('for anti in [False,True]:')[0];ns={'__file__':str(P/'spinor_probe.py')};exec(compile(code,'spinor_definitions','exec'),ns)
C=ns['C'];spin=ns['spin_sum'];start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol and math.isfinite(error))))
model=C.FormFactors()
for anti in [False,True]:
 for ml,E in [(C.V.ME,37.6),(C.V.MMU,218.4)]:
  ch=C.V.Channel(C.V.MP if anti else C.V.MN,C.V.MN if anti else C.V.MP,ml,anti);a,b=ch.initial_mass,ch.final_mass;M=(a+b)/2;s=a*a+2*a*E;root=math.sqrt(s);pi=(s-a*a)/(2*root);pf=math.sqrt((s-(b+ml)**2)*(s-(b-ml)**2))/(2*root);El=(s+ml*ml-b*b)/(2*root);Eb=(s+b*b-ml*ml)/(2*root);center=ml*ml-2*pi*El;half=2*pi*pf
  def integrand(t):
   c=(t-center)/half;st=math.sqrt(max(0.,1-c*c));p=dict(k=np.array([pi,0.,0.,pi]),pa=np.array([root-pi,0.,0.,-pi]),l=np.array([El,pf*st,0.,pf*c]),pb=np.array([Eb,-pf*st,0.,-pf*c]),s=s,pi=pi,pf=pf)
   ff=model.values(t,M);amp,_=spin(ch,p,ff)
   return amp/(64*math.pi*s*pi*pi)
  result,error=quad(integrand,center-half,center+half,epsabs=1e-24,epsrel=1e-10);cm2=result*C.V.HBARC**2*1e-26;author=C.total(ch,E,model,order=64)
  gate(f'independent t phase/spinor total anti{anti}m{ml}',cm2/author-1,3e-11);rows.append(dict(anti=anti,m=ml,E=E,t_integrated_cm2=cm2,author_cos64_cm2=author,quad_relative_error=error/result))
# Same-kinematics equal-mass limit directly exposes leptonic crossing signs.
parts=[]
for anti in [False,True]:
 ch=C.V.Channel(939.,939.,C.V.MMU,anti);p=C.V.momenta(ch,187.4,-.27);q=p['pb']-p['pa'];t=q[0]**2-q[1:]@q[1:];ff=model.values(t,939.);amp,decomp=spin(ch,p,ff);parts.append(decomp)
for key in parts[0]:
 expected=-1 if key in ['VA','AF']else 1;scale=max(abs(parts[0][key]),abs(parts[1][key]),1e-20)
 if key in ['VP','FP']:scale=max(abs(x)for x in parts[0].values())
 gate('explicit spinor equalmass crossing '+key,(parts[1][key]-expected*parts[0][key])/scale,3e-12)
for anti in [False,True]:
 ch=C.V.Channel(939.,939.,0.,anti);p=C.V.momenta(ch,173.1,.37);q=p['pb']-p['pa'];ff=model.values(q[0]**2-q[1:]@q[1:],939.)
 full,parts0=spin(ch,p,ff);off,_=spin(ch,p,(ff[0],ff[1],ff[2],0.));gate('independent massless PS zero '+str(anti),full/off-1,3e-12)
r=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,equalmass_spinor_parts=parts,runtime_seconds=time.monotonic()-start);(P/'normalization_results.json').write_text(json.dumps(r,indent=2)+'\n');print(len(checks),r['passed'],r['runtime_seconds']);print([c for c in checks if not c['passed']])
# Disable the definitions-only probe's atexit partial saver.
import atexit
atexit.unregister(ns['save'])
