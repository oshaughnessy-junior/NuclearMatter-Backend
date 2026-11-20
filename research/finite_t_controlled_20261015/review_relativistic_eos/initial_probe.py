"""Independent momentum quadrature/root reference; no author test replay."""
from pathlib import Path
import sys,time,json,math,hashlib,atexit
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit,kve
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'relativistic_free_eos'))
import free_baryons as A
start=time.monotonic();checks=[];rows=[];phase='start';files=[ROOT/'relativistic_free_eos'/f for f in ['free_baryons.py','results.json','extra_results.json','initial_antiparticle_exclusion_results.json','initial_cold_roundoff_results.json']]+[ROOT/'thermodynamics/lepton_reference.py'];hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files};C=1/(math.pi**2*A.HBARC**3)
def save():
 end={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,runtime_seconds=time.monotonic()-start,checks=checks,rows=rows,source_sha256=hashes,end_source_sha256=end),indent=2,default=lambda x:x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def independent(T,mu,m,pairs=False):
 # Momentum variable with separate occupation of particles and antiparticles.
 end=math.sqrt((max(m,abs(mu))+55*T)**2-m*m);cuts=[0.,end]
 for s in ([mu,-mu]if pairs else [mu]):
  for k in [-16,-5,0,5,16]:
   E=s+k*T
   if m<E<math.hypot(m,end):cuts.append(math.sqrt(E*E-m*m))
 def integrand(p):
  E=math.hypot(m,p);f=expit((mu-E)/T);b=expit((-mu-E)/T)if pairs else 0.;tot=f+b;net=f-b;wf=f*expit((E-mu)/T);wb=b*expit((E+mu)/T);x=(E-mu)/T;y=(E+mu)/T
  def ent(x):return np.logaddexp(0.,-abs(x))+abs(x)*expit(-abs(x))
  return C*p*p*np.array([net,E*tot,p*p/(3*E)*tot,ent(x)+(ent(y)if pairs else 0.),(wf+wb)/T,((E-mu)*wf-(E+mu)*wb)/T**2,((E-mu)**2*wf+(E+mu)**2*wb)/T**2])
 out=np.zeros(7)
 for lo,hi in zip(sorted(set(cuts))[:-1],sorted(set(cuts))[1:]):out+=quad_vec(integrand,lo,hi,epsabs=1e-17,epsrel=2e-11)[0]
 # Cv at fixed particle or net density: U_T|mu - U_mu n_T/n_mu,
 # U_T|mu=T*s_T+mu*n_T; covariance form in both cases.
 out=np.r_[out,out[6]-T*out[5]**2/out[4]]
 return out

def root(T,n,m,pairs=False):
 def fn(mu):return independent(T,mu,m,pairs)[0]-n
 lo=-max(m,abs(n)**(1/3)*A.HBARC*10)-100*T if pairs else m-100*T
 hi=max(m,abs(n)**(1/3)*A.HBARC*10)+100*T
 mu=brentq(fn,lo,hi,xtol=2e-10,rtol=1e-13)
 return mu,independent(T,mu,m,pairs)
for T,n,m in [(3.7,.0034,A.MN),(.19,.021,A.MP)]:
 phase=f'baryon {T}';s=A.state_density(T,n,m);v=independent(T,s.chemical_potential,m)
 gate(phase+' forward density before derivatives',v[0]/n-1,3e-10)
 mu,z=root(T,n,m);gate(phase+' independent momentum density root',mu-s.chemical_potential,3e-8)
 for label,a,b in [('energy',s.energy_density,v[1]),('pressure',s.pressure,v[2]),('entropy',s.entropy_density,v[3]),('susceptibility',s.dn_dmu_fixed_temperature,v[4]),('Cv',s.cv_fixed_density,v[7]),('n_T',s.dn_dtemperature_fixed_mu,v[5])]:gate(phase+' '+label,a/b-1,3e-9)
 # FD ladder of independent momentum quadratures, keeping total mu fixed.
 for h in [T*2e-3,T*1e-3]:
  up=independent(T+h,s.chemical_potential,m);dn=independent(T-h,s.chemical_potential,m)
  gate(phase+f' independent FD n_T h{h}',(up[0]-dn[0])/(2*h)/s.dn_dtemperature_fixed_mu-1,2e-5)
 rows.append(dict(kind='species',T=T,n=n,m=m,author=s.asdict(),momentum=v,independent_mu=mu));save()
# New assembled composition; independent roots for all four species first.
T,n,Ye,Ymu=3.3,.0023,.118,.047;phase='matter';s=A.matter_state(T,n,Ye,Ymu);specs=[(A.MN,n*(1-Ye-Ymu),False,s.neutron),(A.MP,n*(Ye+Ymu),False,s.proton),(A.leptons.ELECTRON_MASS,n*Ye,True,s.electron),(A.leptons.MUON_MASS,n*Ymu,True,s.muon)];vs=[]
for i,(m,d,pair,st)in enumerate(specs):
 mu,v=root(T,d,m,pair);vs.append(v);gate(f'matter species{i} root',mu-st.chemical_potential,5e-8);gate(f'matter species{i} forward density',v[0]/d-1,3e-10)
for label,attr,j in [('energy','energy_density',1),('pressure','pressure',2),('entropy','entropy_density',3),('Cv','cv_fixed_composition',7)]:gate('matter independent sum '+label,getattr(s,attr)/sum(v[j]for v in vs)-1,5e-9)
gate('matter charge closure',(s.electron.net_density+s.muon.net_density)/s.proton.density-1,3e-10)
gate('physical rest mass difference',s.neutron.mass-s.proton.mass-(A.MN-A.MP),1e-12)
der=A.matter_derivatives(s)
for field,val in [('temperature',T),('nB',n),('Ye',Ye),('Ymu',Ymu)]:
 errors=[]
 for frac in [2e-3,1e-3]:
  h=(val if field in ['temperature','nB']else 1)*frac
  args=dict(T=T,nB=n,Ye=Ye,Ymu=Ymu);key='T'if field=='temperature'else field;args[key]=val+h;up=A.matter_state(**args);args[key]=val-h;dn=A.matter_state(**args);fd=(up.energy_density/up.nB-dn.energy_density/dn.nB)/(2*h);target=der[{'temperature':'u_T','nB':'u_nB','Ye':'u_Ye','Ymu':'u_Ymu'}[field]];err=fd/target-1;gate('matter FD '+field+str(frac),err,2e-5);errors.append(err)
 rows.append(dict(kind='matter derivative',coordinate=field,errors=errors));save()
recovered=A.temperature_from_energy(sum(v[1]for v in vs),n,Ye,Ymu,T*1.13);gate('temperature inversion of independent total U',recovered/T-1,2e-9)
# Extremely suppressed anti diagnostic from a distinct scaled Bessel construction.
s=A.state_density(.0013,2e-8,A.MN);expected=math.log(A.MN**2*.0013/(math.pi**2*A.HBARC**3))+math.log(kve(2,A.MN/.0013))-(A.MN+s.chemical_potential)/.0013
gate('new cold antiparticle finite log',s.log_antiparticle_density-expected,3e-9);gate('underflow representation flag',float(s.antiparticle_density_representable or s.antiparticle_density!=0),0.)
for label,fn in [('original T37 anti exclusion',lambda:A.state_mu(37,A.MP-18*37,A.MP)),('zero neutron',lambda:A.matter_state(2,.001,.6,.4)),('zero proton',lambda:A.matter_state(2,.001,0,0)),('zero species',lambda:A.state_density(2,0)),('NaN density',lambda:A.state_density(2,float('nan')))]:
 try:fn();rejected=False
 except (ValueError,ArithmeticError):rejected=True
 gate(label,float(not rejected),0.)
rows.append(dict(kind='assembled state',author=s.asdict(),T=T,nB=n,Ye=Ye,Ymu=Ymu,independent_total_energy=sum(v[1]for v in vs),recovered_temperature=recovered))
phase='complete';save();print(sum(c['passed']for c in checks),len(checks),time.monotonic()-start)
