"""Fresh total coefficient: outgoing logmomentum/q with exact dc Jacobian.

Inner scalar phase primitive is shared; independent delta/spinor points are in probe.py.
"""
from pathlib import Path
import importlib.util,sys,math,time,json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
P=Path(__file__).resolve().parent;A=P.parent/'hot_ec_opacity_spectrum';sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('hotEC_review_outer',A/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
def q_integral(E,order=8,qorder=12,tail=75.,outertail=40.,budget=105.):
 start=time.monotonic();b,_=S.matched();me=S.C.ME;mid=E+S.C.MN-S.C.MP;upper=mid+outertail*b.T
 # Regular logarithmic outgoing momentum, not authors sqrt(Eelectron−me).
 cuts=sorted({me,upper,math.hypot(E,me),mid}|{v for h in[-20,-12,-6,-3,-1.5,-.7,-.3,-.1,.1,.3,.7,1.5,3,6,12,20]if me<(v:=mid+h*b.T)<upper})
 x,w=leggauss(order);xx,ww=leggauss(qorder);logs={'j':[],'chi':[]};count=0;timelike=0
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  lp=lambda el:math.log1p(math.sqrt((el-me)*(el+me))/b.T)
  a,z=lp(lo),lp(hi)
  for t,wt in zip((a+z)/2+(z-a)*x/2,w*(z-a)/2):
   pe=b.T*math.expm1(t);El=math.hypot(pe,me);dpdt=b.T*math.exp(t);omega=E-El;qmin=abs(E-pe);qmax=E+pe
   qcuts=sorted({qmin,qmax}|{q for q in [abs(omega),math.sqrt(max(0.,omega*omega-(S.C.MN-S.C.MP)**2))]if qmin<q<qmax})
   for qa,qb in zip(qcuts[:-1],qcuts[1:]):
    for q,wq in zip((qa+qb)/2+(qb-qa)*xx/2,ww*(qb-qa)/2):
     if time.monotonic()-start>budget:raise TimeoutError('Alternate outer physical integral budget')
     cos=(E*E+pe*pe-q*q)/(2*E*pe)
     # dEe/dpe=pe/Ee, |dc/dq|=q/(Enu pe).
     weight=wt*dpdt*wq*q/(E*El)
     r=S.differential(b,E,float(El),float(cos),energy_order=20,phi_order=6,tail=tail);count+=1;timelike+=q<abs(omega)
     for k in logs:
      if r[k]['log']is not None:logs[k].append(math.log(weight)+r[k]['log'])
 return dict(j=math.exp(float(logsumexp(logs['j']))),chi=math.exp(float(logsumexp(logs['chi']))),controls=dict(E=E,order=order,qorder=qorder,tail=tail,outertail=outertail,energy_order=20,phi_order=6),calls=count,timelike_calls=int(timelike),seconds=time.monotonic()-start)
def run():
 start=time.monotonic();E=7.3;checks=[];rows=[]
 def save(): (P/'outer_partial.json').write_text(json.dumps(dict(rows=rows,checks=checks,seconds=time.monotonic()-start),indent=2)+'\n')
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(abs(e)<=t)))
 refs=[]
 for control in [(8,12,12,4,50.,30.),(12,20,16,8,70.,40.)]:
  n,a,e,p,t,ot=control;r=S.integrate(E,lepton_order=n,angle_order=a,energy_order=e,phi_order=p,tail=t,outer_tail=ot);refs.append(r);rows.append(dict(kind='authornewE',result=r));save()
 for control in [(6,10,65.,35.),(10,16,85.,45.)]:
  n,q,t,ot=control;r=q_integral(E,n,q,t,ot);rows.append(dict(kind='alternate_logmomentum_q',result=r));save()
 fine=rows[-1]['result'];coarse=rows[-2]['result']
 for key in ['j','chi']:
  gate('newauthorcontrols '+key,refs[0][key]['value']/refs[1][key]['value']-1,3e-6)
  gate('alternateouter refinement '+key,coarse[key]/fine[key]-1,3e-6)
  gate('alternateouter newE absolute '+key,fine[key]/refs[1][key]['value']-1,3e-6)
 out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/'outer_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
if __name__=='__main__':run()
