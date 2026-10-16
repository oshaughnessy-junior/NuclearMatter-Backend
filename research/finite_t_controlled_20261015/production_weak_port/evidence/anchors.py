from pathlib import Path
import sys,math,json,importlib.util,time
import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parents[1];R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference import _weak_current as C,_weak_decay as D
from nmbackend.reference._weak_quadrature import log_fd
sys.path.insert(0,str(R/'muon_vacuum'));sys.path.insert(0,str(R/'muon_full_current'))
import current as OLD
checks=[];rows=[]
def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol)))
def cross(channel,energy,order):
 anti=channel=='PC';ma,mb=(C.MP,C.MN)if anti else(C.MN,C.MP);x,w=leggauss(order);s=ma*ma+2*ma*energy;root=math.sqrt(s);pi=ma*energy/root;pf=math.sqrt((s-(mb+C.ME)**2)*(s-(mb-C.ME)**2))/(2*root);le=(s+C.ME*C.ME-mb*mb)/(2*root)
 k=np.tile([energy,0,0,energy],(order,1));pa=np.tile([ma,0,0,0],(order,1));beta=np.tile([0,0,energy/(ma+energy)],(order,1));direction=np.column_stack([np.sqrt(1-x*x),np.zeros(order),x]);l=C.boost(np.column_stack([np.full(order,le),pf*direction]),beta);pb=C.boost(np.column_stack([np.full(order,root-le),-pf*direction]),beta);pn,pp=(pb,pa)if anti else(pa,pb)
 # Avg4 capture-emission amplitude -> avg2 incoming baryon absorption amplitude.
 amp=2*C.scalar_capture(channel,k,pn,l,pp)
 return float(w@(amp/(32*math.pi*s)*pf/pi*C.HC**2*1e-26))
for channel in['EC','PC']:
 ch=OLD.V.Channel(C.MP,C.MN,C.ME,True)if channel=='PC'else OLD.V.Channel(C.MN,C.MP,C.ME)
 for energy in[5.,20.]:
  a=cross(channel,energy,16);b=cross(channel,energy,24);ref=OLD.total(ch,energy,order=24,method='trace');check(f'{channel}{energy}CMcross_refine',a/b-1,3e-10);check(f'{channel}{energy}absolute_primary_trace_cm2',b/ref-1,3e-10);rows.append(dict(channel=channel,Enu=energy,sigma_cm2=b,independent_source_trace=ref))
rest=D.rest_events(20,'pair');alt=D.rest_events(28,'electron');gamma=sum(rest['weight']);gnu=float(rest['weight']@rest['neutrino'][:,0]);check('vacuum beta absolute pair/electron',gamma/sum(alt['weight'])-1,3e-6)
# Compare independent old complete primary event implementation, not a lifetime calibration.
sys.path.insert(0,str(R/'relativistic_crossed_contract'));import beta as B
old=B.electron_integral(28);check('vacuum beta source_absolute',gamma/old['rate']-1,3e-6)
T=2.;mu=919.8287578100662
# Original momentum adaptive independent particle and proper densities.
def density_integrand(p,proper=False):
 en=math.hypot(C.MN,p);return p*p/(math.pi**2*C.HC**3)*math.exp(float(log_fd((mu-en)/T)))*(C.MN/en if proper else 1.)
n=quad(lambda p:density_integrand(p),0,math.inf,epsabs=1e-23,epsrel=1e-11)[0];proper=quad(lambda p:density_integrand(p,True),0,math.inf,epsabs=1e-23,epsrel=1e-11)[0]
rot=D.orient(rest,12,12);x,w=leggauss(32);pnmax=math.sqrt(80*T*(80*T+2*C.MN));p=(x+1)*pnmax/2;wp=w*pnmax/2;rate=energy=dn=0.
for momentum,weight in zip(p,wp):
 en=math.hypot(C.MN,momentum);g=en/C.MN;v=momentum/en;measure=momentum**2*weight/(math.pi**2*C.HC**3)*math.exp(float(log_fd((mu-en)/T)));dn+=measure
 # Actual phase-integrated thermal unblocked event weight and boosted nu energy.
 event=measure/g*rot['weight'];rate+=float(sum(event));energy+=float(event@(g*(rot['neutrino'][:,0]+v*rot['neutrino'][:,3])))
check('unblocked FDdensity quad',dn/n-1,3e-10);check('thermal proper lifetime',rate/(proper*gamma)-1,3e-6);check('thermal Doppler neutrino energy',energy/(n*gnu)-1,3e-6)
output=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),cross_sections=rows,rest_rate=gamma,rest_lifetime=1/gamma,unblocked_actual=dict(rate=rate,neutrino_energy=energy),independent_density=n,independent_proper_density=proper,model='same declared full tree current; no empirical lifetime or uncertainty estimate')
(P/'evidence/anchors.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(passed=output['passed'],failed=output['failed'],rest_rate=gamma)))
