"""Independent momentum/grandpotential and nested-root oracle; no NMB imports."""
from pathlib import Path
import math,json,time,sys
import numpy as np
from scipy.integrate import quad,quad_vec
from scipy.optimize import brentq
from scipy.special import expit,log_expit
P=Path(__file__).resolve().parent;HC=197.3269804;M=939/HC;Cs=11.785;Co=7.148;Cr=4.410;BSC=.002947;CSC=-.001070
NAMES=['net','scalar','energy','pressure','entropy','chi','n_t','B','Q','rho_t','s_t','grandpotential','particle','anti']
def panels(t,nu,m,tail):
 upper=max(m,abs(nu))+tail*t;pu=math.sqrt((upper-m)*(upper+m));ps=[0.,pu]
 for e in [abs(nu)+a*t for a in[-36,-11,-3,0,3,11,36]]+[m+a*t for a in[.3,1.7,5.1,15]]:
  if m<e<upper:ps.append(math.sqrt((e-m)*(e+m)))
 return sorted(set(ps))
def number(t,nu,m,tail=90):
 ps=panels(t,nu,m,tail)
 return sum(quad(lambda p:p*p*(expit((nu-math.hypot(p,m))/t)-expit((-nu-math.hypot(p,m))/t))/math.pi**2,l,h,epsabs=1e-15,epsrel=8e-12,limit=150)[0]for l,h in zip(ps[:-1],ps[1:]))
def gas(t,nu,m,tail=90):
 def fun(p):
  E=math.hypot(p,m);u=(E-nu)/t;v=(E+nu)/t;f=expit(-u);fb=expit(-v);w=f*expit(u);wb=fb*expit(v);rs=m/E
  return p*p/math.pi**2*np.array([f-fb,rs*(f+fb),E*(f+fb),p*p/(3*E)*(f+fb),u*f+np.logaddexp(0,-u)+v*fb+np.logaddexp(0,-v),(w+wb)/t,((E-nu)*w-(E+nu)*wb)/t**2,rs*(w-wb)/t,p*p/E**3*(f+fb)-rs*rs*(w+wb)/t,rs*((E-nu)*w+(E+nu)*wb)/t**2,((E-nu)**2*w+(E+nu)**2*wb)/t**3,-t*(np.logaddexp(0,-u)+np.logaddexp(0,-v)),f,fb])
 ps=panels(t,nu,m,tail);vals=sum((quad_vec(fun,l,h,epsabs=2e-14,epsrel=8e-12)[0]for l,h in zip(ps[:-1],ps[1:])),np.zeros(len(NAMES)));out=dict(zip(NAMES,map(float,vals)))
 logscale=float(log_expit(-(m+nu)/t))
 bar_scaled=sum(quad(lambda p:p*p/math.pi**2*math.exp(float(log_expit(-(math.hypot(p,m)+nu)/t))-logscale),l,h,epsabs=1e-16,epsrel=8e-12)[0]for l,h in zip(ps[:-1],ps[1:]))
 out['log_antiparticle']=logscale+math.log(bar_scaled);out['pair_Cv_fixed_net']=t*(out['s_t']-out['n_t']**2/out['chi']);return out

def nu_density(t,density,m,tail=90):
 kf=(3*math.pi**2*density)**(1/3);hi=math.sqrt(m*m+kf*kf)+10*t
 return brentq(lambda nu:number(t,nu,m,tail)-density,0.,hi,xtol=1e-13,rtol=1e-14)
def state(n,T,Y,tail=90,intervals=73):
 start=time.monotonic();t=T/HC;nn=n*(1-Y);pp=n*Y;calls=[]
 def densities(phi):
  mass=M-phi;nnu=nu_density(t,nn,mass,tail);pnu=nu_density(t,pp,mass,tail)
  return mass,nnu,pnu
 def residual(phi):
  if time.monotonic()-start>108:raise TimeoutError('Pure nested-root oracle budget')
  mass,nu,pnu=densities(phi)
  def scalar(v):
   ps=panels(t,v,mass,tail)
   return sum(quad(lambda p:p*p/math.pi**2*mass/math.hypot(p,mass)*(expit((v-math.hypot(p,mass))/t)+expit((-v-math.hypot(p,mass))/t)),l,h,epsabs=2e-15,epsrel=8e-12)[0]for l,h in zip(ps[:-1],ps[1:]))
  val=phi/Cs+BSC*M*phi*phi+CSC*phi**3-scalar(nu)-scalar(pnu);calls.append([phi,val]);return val
 grid=np.linspace(0,.985*M,intervals+1);f=[residual(x)for x in grid];brackets=[(float(l),float(h))for l,h,a,b in zip(grid[:-1],grid[1:],f[:-1],f[1:])if a<0<=b]
 if len(brackets)!=1:raise ArithmeticError('Pureoracleambiguouscanonicalupcrossing'+repr(brackets))
 phi=brentq(residual,*brackets[0],xtol=2e-13,rtol=1e-14);mass,nu,pnu=densities(phi);a=gas(t,nu,mass,tail);b=gas(t,pnu,mass,tail);W=Co*n;rho=Cr*(pp-nn)/2;vn=W-rho/2;vp=W+rho/2;U=phi*phi/(2*Cs)+BSC*M*phi**3/3+CSC*phi**4/4;V=Co*n*n/2+Cr*(pp-nn)**2/8
 E=(a['energy']+b['energy']+U+V)*HC;pr=(a['pressure']+b['pressure']-U+V)*HC;s=a['entropy']+b['entropy'];D=1/Cs+2*BSC*M*phi+3*CSC*phi*phi+sum(z['Q']+z['B']**2/z['chi']for z in[a,b]);avec=np.array([z['B']/z['chi']for z in[a,b]]);H=(np.diag([1/a['chi'],1/b['chi']])-np.outer(avec,avec)/D+np.array([[Co+Cr/4,Co-Cr/4],[Co-Cr/4,Co+Cr/4]]))*HC;K=sum(z['rho_t']-z['B']*z['n_t']/z['chi']for z in[a,b]);cv=t*(sum(z['s_t']-z['n_t']**2/z['chi']for z in[a,b])+K*K/D)
 return dict(nB=n,T=T,Yp=Y,phi=phi*HC,W=W*HC,R=rho*HC,mD=mass*HC,nu_n=nu*HC,nu_p=pnu*HC,mu_n=(nu+vn)*HC,mu_p=(pnu+vp)*HC,energy=E,pressure=pr,entropy=s,free_energy=E-T*s,Cv=cv,D=D,H=H.tolist(),H_eigenvalues=np.linalg.eigvalsh(H).tolist(),scalar_residual=residual(phi),number_residuals=[a['net']/nn-1,b['net']/pp-1],neutron=a,proton=b,branch_brackets=brackets,scalar_calls=calls,tail=tail,seconds=time.monotonic()-start)
if __name__=='__main__':
 lock=json.loads((P/'PREPARATION_LOCK.json').read_text());i=int(sys.argv[1]);point=(lock['heldouts_nB_T_Yp']+[lock['spinodal_diagnostic_nB_T_Yp']])[i];z=state(*point);(P/f'oracle_state_{i}.json').write_text(json.dumps(z,indent=2)+'\n');print('PUREORACLE',point,'seconds',z['seconds'],'H',z['H_eigenvalues'],'residual',z['number_residuals'],z['scalar_residual'])
