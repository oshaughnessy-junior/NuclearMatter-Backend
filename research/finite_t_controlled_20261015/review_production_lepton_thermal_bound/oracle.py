"""Independent scaled momentum/Bessel thermal and cold-hole bounds; no candidate import."""
from pathlib import Path
import math,json,sys,datetime,hashlib
import numpy as np
from scipy.special import kve
from scipy.integrate import quad
from scipy.optimize import brentq
P=Path(__file__).resolve().parent
ME=.510998950;MMU=105.6583755;HC=197.3269804
MIN=float(np.nextafter(0.,1.));LOG_HALF=math.log(MIN)-math.log(2.)
def state(v,alpha,m=ME):
 T=m/v;N=m*m*T/(math.pi**2*HC**3)*math.exp(-v)*kve(2,v);n=2*N*math.sinh(alpha);mean=3*T+m*kve(1,v)/kve(2,v);energy=2*N*math.cosh(alpha)*mean
 pfd=HC*(3*math.pi**2*abs(n))**(1/3)/T;xc=pfd*pfd/(math.hypot(pfd,v)+v)
 logbound=(math.log(4/15)+.5*math.log(xc+2*v)+math.log(v+xc)+2.5*math.log(xc)+4*math.log(T)-math.log(math.pi**2*HC**3))if xc else -math.inf
 # Finite scaled t=x/xc integration keeps a positive kernel of order v^1.5.
 norm=quad(lambda t:math.sqrt(t)*math.sqrt(2*v+xc*t)*(v+xc*t)*(1-t),0.,1.,epsabs=1e-13,epsrel=5e-13)[0]if xc else 0.
 loghole=(math.log(norm)+2.5*math.log(xc)+4*math.log(T)-math.log(math.pi**2*HC**3))if xc else -math.inf
 # Independently ordered lepton momentum k=p/sqrt(mT), FD denominators retained.
 def ints(power,net):
  def f(k):
   E=math.hypot(m,k*math.sqrt(m*T));x=k*k/(math.sqrt(1+k*k/v)+1)
   fminus=math.exp(alpha-x)/(1+math.exp(-v+alpha-x));fplus=math.exp(-alpha-x)/(1+math.exp(-v-alpha-x))
   return k*k*(fminus-fplus if net else fminus+fplus)*E**power
  return (m*T)**1.5*math.exp(-v)/(math.pi**2*HC**3)*quad(f,0.,20.,epsabs=1e-11,epsrel=3e-13)[0]
 nm=ints(0,True);em=ints(1,False)
 return dict(v=v,alpha=alpha,mass=m,T=T,mu=alpha*T,number=n,energy=energy,thermal=energy-m*abs(n),momentum_number=nm,momentum_energy=em,momentum_thermal=em-m*abs(nm),xc=xc,log_upper_hole_physical=logbound,log_hole_scaled_integral=loghole,hole_float=math.exp(loghole)if loghole>-math.inf else 0.,skip_bound=logbound<LOG_HALF,log_halfminsubnormal=LOG_HALF,FD_relative_correction_bound=math.exp(-v+abs(alpha))/(1-math.exp(-v+abs(alpha))))
def prepare():
 states=[]
 for m in[ME,MMU]:
  root=brentq(lambda v:state(v,.53,m)['log_upper_hole_physical']-LOG_HALF,350.,550.,xtol=2e-12)
  for delta in[-2.,-.2,.2,2.]:
   v=brentq(lambda v:state(v,.53,m)['log_upper_hole_physical']-LOG_HALF-delta,root-5,root+5,xtol=2e-12)
   states.append(state(v,.53,m))
 for alpha in[-.53,.53]:states.append(state(600.,alpha))
 lock=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_imported=False,known_old_failure=[dict(v=600.,alpha=a,mass=ME)for a in[-.53,.53]],nearbound_selection='Independent Bessel/momentum roots at log rigorous hole bound=log(halfminsubnormal)+[-2,-.2,.2,2], electron and muon; fixedbeforecandidatefreeze',states=states,gates=dict(state_and_thermal=3e-10,bessel_vs_scaled_momentum=3e-11),additional_valid_states=[dict(T=.731,mu=-1.193,mass=ME),dict(T=15.7,mu=87.4,mass=MMU),dict(T=.012,mu=153.2,mass=MMU)],scope='Oldfailure+near-bound componentpolicy, preserving total heat and exactold-statefields; minimumAPIstatic/helper/import versus realminimumruntime separately.')
 (P/'VALIDATION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(dict(nearbound=[dict(m=s['mass'],v=s['v'],offset=s['log_upper_hole_physical']-LOG_HALF,hole=s['hole_float'])for s in states[:8]],known=[s['thermal']for s in states[8:]]),indent=2))
if __name__=='__main__':prepare()
