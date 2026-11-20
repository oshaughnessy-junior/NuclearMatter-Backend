"""Independent original momentum FD oracle; no production candidate imports."""
from pathlib import Path
import math,json,datetime
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,log_expit,kve
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;MN=939.565421;MP=938.272088;HC=197.3269804;C=1/(math.pi**2*HC**3)
def momentum(T,mu,m,tail=90.):
 eta=(mu-m)/T;logscale=float(log_expit(eta));scale=math.exp(logscale)
 if not scale:raise ArithmeticError('Oracle ordinarypopulation unrepresented')
 upper=max(0.,mu-m)+tail*T;pf=lambda K:math.sqrt(K*(K+2*m));pmax=pf(upper);p0=math.sqrt(m*T)
 cuts=sorted(set([0.,pmax]+[pf(K)for a in[-40,-12,-4,0,4,12,40]if 0<(K:=mu-m+a*T)<upper]+[x*p0 for x in[.5,1,2,4]if x*p0<pmax]))
 def fun(p,k):
  E=math.hypot(p,m);K=p*p/(E+m);z=K/T;logf=float(log_expit(eta-z));fs=math.exp(logf-logscale);f=math.exp(logf);w=fs*expit(z-eta)
  if k==0:v=fs
  elif k==1:v=fs*z
  elif k==2:v=fs*p*p/(3*E*T)
  elif k==3:
   # Stable exact entropy, separately scaled before dimensional restoration.
   a=eta-z;soft=lambda x:max(0.,x)+math.log1p(math.exp(-abs(x)))
   v=fs*soft(-a)+math.exp(float(log_expit(-a))-logscale)*soft(a)
  elif k==4:v=w
  elif k==5:v=w*(z-eta)
  else:v=w*(z-eta)**2
  return p*p*v
 vals=[sum(quad(lambda p:fun(p,k),lo,hi,epsabs=0.,epsrel=3e-12,limit=250)[0]for lo,hi in zip(cuts[:-1],cuts[1:]))*C*scale for k in range(7)]
 n,kin,pres,ent,a0,a1,a2=vals;kin*=T;pres*=T
 return dict(T=T,mu=mu,mass=m,number=n,kinetic=kin,energy=m*n+kin,pressure=pres,entropy=ent,free_energy=mu*n-pres,chi=a0/T,n_T=a1/T,mu_T=-a1/a0,cv=a2-a1*a1/a0)
def density(T,n,m):
 coldK=(HC*(3*math.pi**2*n)**(1/3))**2
 coldK=coldK/(math.sqrt(coldK+m*m)+m)
 eta=brentq(lambda e:momentum(T,m+e*T,m)['number']/n-1,-70.,max(10.,coldK/T+10),xtol=2e-12)
 return momentum(T,m+eta*T,m)
def fugacity(T,mu,m,terms=16):
 eta=(mu-m)/T;v=m/T
 n=m*m*T*C*sum((-1.)**(j+1)*math.exp(j*eta)*kve(2,j*v)/j for j in range(1,terms+1))
 p=m*m*T*T*C*sum((-1.)**(j+1)*math.exp(j*eta)*kve(2,j*v)/j**2 for j in range(1,terms+1))
 e=m*m*T*T*C*sum((-1.)**(j+1)*math.exp(j*eta)*(3*kve(2,j*v)/j**2+v*kve(1,j*v)/j)for j in range(1,terms+1))
 return dict(number=n,pressure=p,energy=e)
def prepare():
 states=[]
 for T,n,m in[(2.,7e-8,MN),(2.,3e-8,MP),(1.37,4.6e-5,MN),(.041,.16,MN),(.073,.27,MP),(3.4,.12,MN)]:states.append(dict(kind='density',T=T,n=n,mass=m))
 for T,eta,m in[(.23,-31.7,MN),(1.9,-15.3,MP),(2.71,-4.2,MN)]:states.append(dict(kind='mu',T=T,mu=m+eta*T,mass=m))
 lock=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_imported=False,states=states,forward_gate=3e-10,response_gate=2e-8,derivative_gate=2e-6,source_scope='ParticleONLYspin2 idealrelativistic FD gas, positiveparticle density, physicalNMBmasses,totalrestinclusive mu. Noantibaryonpopulation/pairneutrality/interactions/JAXclaim.',tests=['Forwardoriginalmomentum BEFORE derivatives','Independent16termfugacityBessel ineta<0 states','Coldrelativistic SommerfeldCv andNR limits asdeclaredapproximate limits','Euler/freeenergy identity','Re-solved fixedparticle-density temperatureladder Cv/µT','Fixedmu pressurederivatives and Maxwellresponses','invalid/nonfinite/bool/signeddensity/unrepresentable guards'],density_policy='Density states executed onlyif within actualfrozenpublicpolicy. Omissiondiagnostics andn=0 boundary must bedistinguished from pairnet-numberutility. No silent expansion.')
 (P/'VALIDATION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(lock['locked_utc'])
if __name__=='__main__':prepare()
