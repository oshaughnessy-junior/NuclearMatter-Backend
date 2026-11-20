from pathlib import Path
import sys,json,time,math,hashlib,atexit
import numpy as np
from scipy.integrate import quad,quad_vec
from scipy.special import expit
sys.dont_write_bytecode=True
from lorentz import *
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];phase='start'
def save():(P/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashlib.sha256((P/'lorentz.py').read_bytes()).hexdigest()),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def angular_I(a,b):
 L=math.log1p(1/a);return quad(lambda t:-math.expm1(-t)*(1-b*a*math.expm1(t)),0,L,epsabs=1e-40,epsrel=3e-12)[0]
def angular_nu(p,s):
 v=p/math.hypot(p,ME);a=s.screening_squared/(4*p*p);points=sorted(set([-1.,1.,max(-1.,1-2*a)]));return s.ni_natural*v*2*math.pi*sum(quad(lambda c:(1-c)*dcs(p,c,s.screening_squared,s.Z),lo,hi,epsabs=1e-20,epsrel=2e-11)[0]for lo,hi in zip(points[:-1],points[1:]))
def momentum_K(s,angular=False):
 # Original momentum integral; actual p²v²tau and -df/dE factors.
 hi=max(ME,s.mu)+65*s.T;energies=sorted(set([ME,hi]+[s.mu+k*s.T for k in[-16,-6,0,6,16]if ME<s.mu+k*s.T<hi]));ps=[math.sqrt((e-ME)*(e+ME))for e in energies]
 def integrand(p):
  E=math.hypot(ME,p);y=(E-s.mu)/s.T;v=p/E;tau=1/(angular_nu(p,s)if angular else frequency(p,s));base=p*p*v*v*tau*expit(-y)*expit(y)/(3*math.pi**2*s.T)
  return base*np.array([1.,y,y*y])
 out=np.zeros(3)
 for l,h in zip(ps[:-1],ps[1:]):out+=quad_vec(integrand,l,h,epsabs=1e-18,epsrel=3e-10)[0]
 return out*np.array([1,s.T,s.T*s.T])
for a in [1e-8,1e-4,.13,.9,3.9,4.,70.,1e4,1e8]:
 for b in [0.,.37,1.]:gate(f'analytic Coulomb integral a{a} v²{b}',coulomb_integral(a,b)/angular_I(a,b)-1,3e-11)
for T,ne,Z in [(.003,1e-14,1),(.003,1e-11,1),(.07,1e-7,2),(.25,1e-4,6)]:
 phase=f'state{T}';s=state(T,ne,Z);r=moments(s);ref=momentum_K(s);fine=moments(s,tail=85,rtol=4e-12)
 for i in range(3):gate(f'{T} independent p vs E K{i}',ref[i]/r['K'][i]-1,3e-8);gate(f'{T} moment tail/rtol K{i}',fine['K'][i]/r['K'][i]-1,3e-9)
 pf=HC*(3*math.pi**2*ne)**(1/3)
 for factor in [.23,1.,3.1]:
  p=pf*factor;gate(f'{T} angular dcs vs analytic nu at{factor}',angular_nu(p,s)/frequency(p,s)-1,3e-10)
 # Actual screening susceptibility: independent momentum-density FD at fixed total mu.
 def n(mu):
  ps=math.sqrt((max(ME,mu)+70*T)**2-ME*ME);cuts=[0.,ps]
  for k in[-10,-3,0,3,10]:
   E=mu+k*T
   if ME<E<math.hypot(ME,ps):cuts.append(math.sqrt((E-ME)*(E+ME)))
  cuts=sorted(set(cuts));return sum(quad(lambda p:p*p*expit((mu-math.hypot(ME,p))/T)/math.pi**2,a,b,epsabs=1e-17,epsrel=1e-11)[0]for a,b in zip(cuts[:-1],cuts[1:]))
 gate(f'{T} forward electron density',n(s.mu)/(ne*HC**3)-1,3e-9)
 for frac in [1e-3,5e-4]:
  h=T*frac;fd=(n(s.mu+h)-n(s.mu-h))/(2*h);gate(f'{T} screening susceptibility FD {frac}',fd/s.screening_susceptibility_natural-1,3e-6)
 K=np.array(r['K']);gate(f'{T} positive thermoelectric covariance',float(K[0]*K[2]-K[1]*K[1]<=0),0.);gate(f'{T} kappa open no greater than Estar0',float(r['kappa_open']>r['kappa_Estar_zero']),0.);gate(f'{T} Onsager/Kelvin sign',r['peltier']-T*r['seebeck'],1e-15);gate(f'{T} positive sigma',float(r['sigma']<=0),0.)
 rows.append(dict(kind='selected_state',result=r,momentum_K=ref,finer=fine));save()
# Fully angular-derived collision time inside the independent momentum quadrature, one state only.
phase='nested physical kernel';s=state(.07,1e-7,2);nested=momentum_K(s,True);base=moments(s)
for i in range(3):gate(f'full angle+momentum chain K{i}',nested[i]/base['K'][i]-1,3e-8)
rows.append(dict(kind='nested_angular_momentum',K=nested))
# Degenerate sequence at fixed electron density; actual susceptibility re-solved each time.
pf=HC*(3*math.pi**2*1e-7)**(1/3);EF=math.hypot(ME,pf);ladder=[]
for frac in [.02,.01,.005]:
 s=state(frac*(EF-ME),1e-7,1);r=moments(s,tail=80,rtol=2e-12);err=r['Lorenz_ratio']/r['WF_limit']-1;de=r['sigma']/r['sigma_Fermi_surface']-1;ladder.append(dict(T_over_TF=frac,T=s.T,WF_relative=err,Drude_relative=de,result=r));save()
gate('degenerate final WF',ladder[-1]['WF_relative'],1e-3);gate('degenerate final Drude',ladder[-1]['Drude_relative'],1e-3)
for a,b in zip(ladder[:-1],ladder[1:]):gate('WF T² convergence',a['WF_relative']/b['WF_relative']/4-1,.015)
rows.append(dict(kind='degenerate_ladder',values=ladder));phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
