from pathlib import Path
from dataclasses import asdict
import sys,json,hashlib,math,importlib.util,ast,time,signal
import numpy as np
from scipy.integrate import quad_vec,quad
from scipy.optimize import brentq
from scipy.special import expit
P=Path(__file__).resolve().parent;C=P.parent;A=C/'free_source_binder';OLD=C.parent/'nmb-finite-t-20261015';sys.dont_write_bytecode=True
signal.alarm(89);start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'MANIFEST.json')=='7b40760f63ce34a9633a4f6d95d65724e51d82a7ab57b6649077ea3a47586fbf'
manifest=json.loads((A/'MANIFEST.json').read_text());assert all(sha(A/n)==h for n,h in manifest.items())
(P/'INITIAL_AUTHOR_PIN.json').write_text(json.dumps(dict(manifest=sha(A/'MANIFEST.json'),files=manifest),indent=2)+'\n')
spec=importlib.util.spec_from_file_location('binder_review_independent_original_momentum',OLD/'review_production_baryon_port/oracle.py');O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
MN,MP,ME,HC=939.56542052,938.27208816,.51099895069,197.3269804;M=1/(math.pi**2*HC**3)
def lepton(T,mu):
 def integrand(p):
  E=math.hypot(p,ME);a=(E-mu)/T;b=(E+mu)/T;f=expit(-a);g=expit(-b);w=f*expit(a);v=g*expit(b)
  return p*p*np.array([f-g,E*(f+g),p*p/(3*E)*(f+g),a*f+np.logaddexp(0,-a)+b*g+np.logaddexp(0,-b),(w+v)/T,((E-mu)*w-(E+mu)*v)/T**2,E*(w-v)/T,E*((E-mu)*w+(E+mu)*v)/T**2,f,g])
 cuts=[0,1,4,10,30,100];v=sum((quad_vec(integrand,T*l,T*h,epsabs=1e-22,epsrel=1e-11)[0]for l,h in zip(cuts,cuts[1:])),np.zeros(10))*M
 n,e,p,s,chi,nt,emu,et,nm,np_=v
 return dict(mu=mu,number=n,energy=e,pressure=p,entropy=s,chi=chi,mu_T=-nt/chi,cv=et-emu*nt/chi,minus=nm,plus=np_)
def independent(T,n,Y,ph):
 nn=O.density(T,(1-Y)*n,MN);pp=O.density(T,Y*n,MP);mu=brentq(lambda mu:lepton(T,mu)['number']-n*Y,0,20,xtol=2e-13);ee=lepton(T,mu)
 eg=math.pi**2*T**4/(15*HC**3)if ph=='trapped_LTE'else 0.
 hn=nn['mu']-T*nn['mu_T'];hp=pp['mu']-T*pp['mu_T'];he=ee['mu']-T*ee['mu_T']
 return dict(energy=nn['energy']+pp['energy']+ee['energy']+eg,pressure=nn['pressure']+pp['pressure']+ee['pressure']+eg/3,entropy=nn['entropy']+pp['entropy']+ee['entropy']+4*eg/(3*T),cv=nn['cv']+pp['cv']+ee['cv']+4*eg/T,eY=n*(hp-hn+he),en=(1-Y)*hn+Y*(hp+he),h=[hn,hp,he],n=nn,p=pp,e=ee)
# Actual independent forward roots computed BEFORE candidate import/comparison.
points=json.loads((P/'LOCK.json').read_text())['heldouts'];oracles=[independent(*x,'absent')for x in points]
(P/'INDEPENDENT_FORWARD.json').write_text(json.dumps(dict(points=points,oracles=oracles,method='Original momentum FD and independent scalar roots before candidate import'),indent=2)+'\n')
sys.path.insert(0,str(A/'staged/src'));from nmbackend.reference import free_matter as F
checks=[];rows=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def rel(name,a,b,gate=3e-8):ck(name,a/b-1,gate)
for i,(T,n,Y)in enumerate(points):
 for ph in ['absent','trapped_LTE']:
  z=oracles[i]if ph=='absent'else independent(T,n,Y,ph);s=F.state_density(T,n,Y,photons=ph);rows.append(dict(point=[T,n,Y],photons=ph,state=asdict(s),oracle=z))
  for label,part,oracle,mass,target in [('n',s.neutron,z['n'],MN,(1-Y)*n),('p',s.proton,z['p'],MP,Y*n),('e',s.electron,z['e'],ME,Y*n)]:
   density=part.net_density if label=='e'else part.particle_density
   rel(f'{i}{ph}{label}number',density,target,1e-9);ck(f'{i}{ph}{label}kineticmu',(part.chemical_potential-mass)-(oracle['mu']-mass),1e-7)
   ck(f'{i}{ph}{label}mass',part.mass-mass,0)
  ck(f'{i}{ph}specific energy thermal scale',(s.energy_density-z['energy'])/n,1e-7)
  for f,k in [('pressure','pressure'),('entropy_density','entropy')]:rel(f'{i}{ph}{f}',getattr(s,f),z[k])
  ck(f'{i}{ph}no muons',0 if s.muons_absent is True else 1,0)
  rel(f'{i}{ph}electron particle',s.electron.particle_density,z['e']['minus']);rel(f'{i}{ph}positron',s.electron.antiparticle_density,z['e']['plus'])
  # Only following unperturbed forward checks: response and derivative chains.
  rel(f'{i}{ph}Cv',s.cv_fixed_nB_Ye,z['cv'],3e-6);ck(f'{i}{ph}eY/n',(s.denergy_dYe_fixed_nB_T-z['eY'])/n,3e-6)
  ck(f'{i}{ph}thermal enthalpy',((s.energy_density+s.pressure)/n-s.denergy_dnB_fixed_T_Ye)-((z['energy']+z['pressure'])/n-z['en']),3e-6)
  for sp,part,bound in [('n',s.neutron,s.anti_bounds[0]),('p',s.proton,s.anti_bounds[1])]:
   mass=part.mass;p0=math.sqrt(mass*T);integral=quad(lambda t:(p0*t)**2*math.exp(-(p0*t)**2/((math.hypot(p0*t,mass)+mass)*T))*p0,0,20,epsabs=0,epsrel=1e-11)[0]
   logn=math.log(M*integral)-(mass+part.chemical_potential)/T
   ck(f'{i}{ph}{sp}anti MB direct momentum log',bound.log_density_upper_bound-logn,3e-9)
   ck(f'{i}{ph}{sp}anti fraction',bound.log_fraction_upper_bound-(logn-math.log(part.particle_density)),3e-9)
   ck(f'{i}{ph}{sp}anti actual admission',0 if bound.log_fraction_upper_bound<=math.log(1e-8)else 1,0)
  eg=math.pi**2*T**4/(15*HC**3)if ph=='trapped_LTE'else 0
  ck(f'{i}{ph}photon energy',s.photon_energy_density-eg,1e-20)
  # Re-solved FD from independent EOS, no candidate energy in the oracle.
  h=2e-4
  for label,args1,args0,den,expect in [('T',(T*(1+h),n,Y),(T*(1-h),n,Y),2*h*T,s.cv_fixed_nB_Ye),('Y',(T,n,Y+h),(T,n,Y-h),2*h,s.denergy_dYe_fixed_nB_T),('n',(T,n*(1+h),Y),(T,n*(1-h),Y),2*h*n,s.denergy_dnB_fixed_T_Ye)]:
   fd=(independent(*args1,ph)['energy']-independent(*args0,ph)['energy'])/den
   if label=='Y':ck(f'{i}{ph}independent FDY per baryon',(fd-expect)/n,3e-6)
   elif label=='n':ck(f'{i}{ph}independent FDn absoluteMeV',fd-expect,3e-6)
   else:rel(f'{i}{ph}independent FDT',fd,expect,3e-5)
assert all(sha(A/n)==h for n,h in manifest.items())
(P/'EOS_RESULTS.json').write_text(json.dumps(dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,origins=dict(free_matter=F.__file__,baryons=F.B.__file__,leptons=F.L.__file__),seconds=time.monotonic()-start),indent=2,allow_nan=False)+'\n')
print('EOS',sum(x['passed']for x in checks),'PASS',sum(not x['passed']for x in checks),'FAIL',time.monotonic()-start)
signal.alarm(0)
