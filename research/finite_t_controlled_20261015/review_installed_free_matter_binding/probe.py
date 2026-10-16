"""Independent momentum EOS binding and entropy-path checks; no author drivers."""
from pathlib import Path
import json,math,hashlib,sys,importlib.util,time,datetime
import numpy as np
from scipy.integrate import quad,quad_vec
from scipy.optimize import brentq
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parent;A=R/'installed_free_matter_binding';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
checks=[];details={};start=time.monotonic()
def ck(name,error,gate):
 checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def save():
 (P/'results.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],details=details,seconds=time.monotonic()-start),indent=2)+'\n')
manifest=json.loads((A/'MANIFEST.json').read_text());before={k:sha(A/k)for k in manifest['files']}
assert before==manifest['files'];assert sha(A/'MANIFEST.json')=='c3db0aacf894cb688de7358d49a0931b6980473c399a50bcddc32049992e4512'
O=load('binding_review_independent_baryon_momentum',R/'review_production_baryon_port/oracle.py')
MN=939.56542052;MP=938.27208816;ME=.51099895069;HC=197.3269804;C=1/(math.pi**2*HC**3)
# Preregister BEFORE loading candidate or computing its comparison values.
lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_imported=False,manifest_sha=sha(A/'MANIFEST.json'),forward_points=[[2,1e-7,.3],[1,1e-7,.3850567101958],[3,1e-7,.5490065797957709],[2,1e-6,.2786277578701],[2.37,2.8e-7,.41]],FD_point=[2.37,2.8e-7,.41],entropy_times=[.125,.25,.75],source_controls=dict(EC=[24,40,10,65],PC=[30,48,11,65],beta=[20,24,9,14,'electron','direct_p',85]),gates=dict(forward=3e-9,analytic_response=3e-8,FD=3e-5,entropy_root=3e-6,work=3e-6,source=3e-6,ledgers=2e-9))
(P/'VALIDATION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
def lepton_mu(T,mu):
 def f(p):
  E=math.hypot(p,ME);a=(E-mu)/T;b=(E+mu)/T;fm=expit(-a);fp=expit(-b);wm=fm*expit(a);wp=fp*expit(b)
  return p*p*np.array([fm-fp,E*(fm+fp),p*p/(3*E)*(fm+fp),a*fm+np.logaddexp(0,-a)+b*fp+np.logaddexp(0,-b),(wm+wp)/T,((E-mu)*wm-(E+mu)*wp)/T**2,E*(wm-wp)/T,E*((E-mu)*wm+(E+mu)*wp)/T**2])
 vals=sum((quad_vec(f,T*l,T*h,epsabs=1e-22,epsrel=1e-11)[0]for l,h in zip([0,1,4,10,30],[1,4,10,30,100])),np.zeros(8))*C
 n,e,p,s,chi,nt,emu,et=vals
 return dict(mu=mu,number=n,energy=e,pressure=p,entropy=s,chi=chi,n_T=nt,mu_T=-nt/chi,cv=et-emu*nt/chi,h=emu/chi)
def independent(T,n,Y,ph=True):
 nn=O.density(T,(1-Y)*n,MN);pp=O.density(T,Y*n,MP)
 mu=brentq(lambda mu:lepton_mu(T,mu)['number']-n*Y,0,20,xtol=2e-13);ee=lepton_mu(T,mu)
 eg=math.pi**2*T**4/(15*HC**3)if ph else 0
 hn=nn['mu']-T*nn['mu_T'];hp=pp['mu']-T*pp['mu_T'];he=ee['h']
 return dict(energy=nn['energy']+pp['energy']+ee['energy']+eg,pressure=nn['pressure']+pp['pressure']+ee['pressure']+eg/3,entropy=nn['entropy']+pp['entropy']+ee['entropy']+4*eg/(3*T),cv=nn['cv']+pp['cv']+ee['cv']+4*eg/T,e_n=(1-Y)*hn+Y*(hp+he),e_Y=n*(hp-hn+he),n=nn,p=pp,e=ee)
M=load('binding_review_frozen_matter',A/'matter.py')
ck('real lepton source hash',0 if sha(Path(M.L.__file__))=='baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719'else 1,0)
ck('real baryon source hash',0 if sha(Path(M.B.__file__))=='f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b'else 1,0)
ck('explicit campaign masses and hbarc',0 if (M.MN,M.MP,M.ME,M.HC)==(MN,MP,ME,HC)else 1,0)
details['module_origins']={x:str(z.__file__)for x,z in [('lepton',M.L),('baryon',M.B),('constants',M.C)]}
for i,(T,n,Y)in enumerate(lock['forward_points']):
 z=independent(T,n,Y);a=M.state(T,n,Y);row=dict(T=T,n=n,Y=Y,oracle=z,candidate=a.asdict());details['forward_'+str(i)]=row
 # Forward fields BEFORE responses.
 for k in ['energy','pressure','entropy']:ck(f'forward{i} {k}',getattr(a,k)/z[k]-1,3e-9)
 for species,oracle,target in [('n',a.neutron,(1-Y)*n),('p',a.proton,Y*n),('e',a.electron,Y*n)]:
  ck(f'forward{i} {species} charge/density',z[species]['number']/target-1,2e-9)
  ck(f'forward{i} {species} chemical potential',(oracle.chemical_potential-z[species]['mu'])/T,3e-9)
 for sp,mass in [('n',MN),('p',MP)]:
  b=O.fugacity(T,z[sp]['mu'],mass)
  for k,kk in [('number','number'),('energy','energy'),('pressure','pressure')]:ck(f'Bessel{i} {sp} {k}',b[k]/z[sp][kk]-1,3e-9)
 for k in ['cv','e_n','e_Y']:ck(f'analytic response{i} {k}',getattr(a,k)/z[k]-1,3e-8)
 # Independently derived enthalpy identity, scale by thermal terms.
 chemT=(1-Y)*a.neutron.dmu_dtemperature_fixed_particle_density+Y*(a.proton.dmu_dtemperature_fixed_particle_density+a.electron.dmu_dtemperature_fixed_net_density)
 lhs=a.energy+a.pressure-n*a.e_n;rhs=T*a.entropy+T*n*chemT
 ck(f'corrected enthalpy{i}',(lhs-rhs)/(T*a.entropy),3e-9)
 save()
T,n,Y=lock['FD_point'];a=M.state(T,n,Y);h=1.7e-4
for k,args1,args0,den in [('cv',(T*(1+h),n,Y),(T*(1-h),n,Y),2*h*T),('e_n',(T,n*(1+h),Y),(T,n*(1-h),Y),2*h*n),('e_Y',(T,n,Y+h),(T,n,Y-h),2*h)]:
 fd=(independent(*args1)['energy']-independent(*args0)['energy'])/den
 ck('fresh independent re-solved energy FD '+k,fd/getattr(a,k)-1,3e-5)
 save()
# Independent entropy roots, no candidate EOS in root and no RK4 RHS reused.
s0=independent(2,1e-7,.3)['entropy']/1e-7;cy=json.loads((A/'cycle_64.json').read_text());root_rows=[]
for t in lock['entropy_times']:
 n=1e-7*math.exp(math.log(2)*math.sin(2*math.pi*t));tr=brentq(lambda temp:independent(temp,n,.3)['entropy']/n-s0,1,3,xtol=2e-10)
 rec=cy['records'][round(64*t)];ck('independent entropy root '+str(t),rec['T']/tr-1,3e-6);root_rows.append(dict(t=t,T=tr,n=n))
details['entropy_roots']=root_rows;save()
# Pressure-work independently integrate over LOG density along constant entropy.
# 8/12 point Gauss; quarters are a single-valued adiabat, not a zero-net cycle.
work=[]
for order in [8,12]:
 x,w=np.polynomial.legendre.leggauss(order);val=0
 for xx,ww in zip(x,w):
  l=(xx+1)*math.log(2)/2;n=1e-7*math.exp(l);tr=brentq(lambda temp:independent(temp,n,.3)['entropy']/n-s0,1.4,2.8,xtol=1e-9)
  val+=ww*independent(tr,n,.3)['pressure']/n*math.log(2)/2
 work.append(val)
du=cy['records'][16]['specific_energy']-cy['records'][0]['specific_energy'];details['work']=dict(Gauss8=work[0],Gauss12=work[1],RK4_delta_u=du)
ck('independent adiabatic pressure work refinement',work[0]/work[1]-1,3e-6);ck('compression work thermal normalization',(du-work[1])/abs(work[1]),3e-6);save()
# Guard checks are binding API tests, not full rectangle qualification.
for args,kw in [((float('nan'),1e-7,.3),{}),((2,1e-7,float('nan')),{}),((2,True,.3),{}),((2,1e-7,.3),dict(photons=1)),((2,1e-7,.3),dict(photons=float('nan')))]:
 try:M.state(*args,**kw);passed=False
 except ValueError:passed=True
 ck('binding rejects '+repr((args,kw)),0 if passed else 1,0)
assert {k:sha(A/k)for k in manifest['files']}==before
ck('all frozen author bytes unchanged',0,0);details['source_pins']=before;save();print('PASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
