"""Independent frozen flat-Dalitz/scalar-current/direct-momentum recipe at HOT inputs."""
from pathlib import Path
import sys,json,time,math,hashlib,atexit,importlib.util
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
R=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'hot_beta_moments'));import hot_moments as A
spec=importlib.util.spec_from_file_location('independent_dalitz',R/'review_crossed_moments/reference.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
files=[x for x in (R/'hot_beta_moments').iterdir()if x.is_file()];pin={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest()for x in files};checks=[];rows=[];phase='start';start=time.monotonic()
def enc(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
def save():
 (P/'results.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,seconds=time.monotonic()-start,source_sha256=pin,end_source_sha256={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest()for x in files}),default=enc,indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
frozen=json.loads((R/'hot_beta_moments/results.json').read_text());stored=[x['result']for x in frozen['rows']if x.get('label')=='Laguerre'][-1]
for T,nB,Y in [(2.,1e-7,.3),(2.4,1.7e-7,.26)]:
 phase='forward roots '+str(T);b,meta=A.K.matched_npe(T,nB,Y);eq=b.mu_n-b.mu_p-b.mu_e
 def density(m,mu,proper=False):
  scale=math.sqrt(2*m*T)
  return quad(lambda y:y*y*expit((mu-math.hypot(m,scale*y))/T)*(m/math.hypot(m,scale*y)if proper else 1.),0,math.sqrt(100),epsabs=1e-20,epsrel=2e-11)[0]*scale**3/(math.pi**2*A.HC**3)
 nd=density(A.K.MN,b.mu_n);np_=density(A.K.MP,b.mu_p);pd=density(A.K.MN,b.mu_n,True)
 gate('independent neutron root '+str(T),nd/(nB*(1-Y))-1,3e-9);gate('independent proton root '+str(T),np_/(nB*Y)-1,3e-9)
 def lepton(sign):return quad(lambda y:y*y*expit((sign*b.mu_e-math.hypot(A.K.ME,T*y))/T),0,100,epsabs=1e-18,epsrel=2e-11)[0]*T**3/(math.pi**2*A.HC**3)
 ne,npos=lepton(1),lepton(-1);gate('independent net charge '+str(T),(ne-npos)/(nB*Y)-1,3e-9)
 refs=[]
 for order,radial,orient,phi in [(20,10,8,12),(28,16,12,16)]:
  phase='Dalitz '+str((T,order));z=D.integrate(b,eq-.8*T,order=order,radial=radial,orient=orient,phi=phi,budget=110);refs.append(z);rows.append(dict(kind='independent',T=T,nB=nB,Y=Y,bath=vars(b),meta=meta,independent_density=nd,independent_proper_density=pd,pair_ratio=npos/ne,result=z));save()
  gate('reference parent forward '+str((T,order)),z['density']/nd-1,3e-9);gate('unblocked proper lifetime '+str((T,order)),z['scenarios']['unblocked']['F'][0]/(pd*z['rest_rate'])-1,3e-8)
  for key in ['neutrino_energy','electron_energy','proton_energy']:
   idx=D.NAMES.index(key);gate('unblocked Doppler '+key,z['scenarios']['unblocked']['F'][idx]/(nd*z['rest_moments'][key])-1,3e-8)
  for name,s in z['scenarios'].items():
   F,B,net=s['F'],s['B'],s['net'];scale=F[1]+B[1];gate('signed energy ledger '+str((T,name)),(net[3]+net[2]-net[4]+net[1])/scale,3e-10);gate('direct material energy '+str((T,name)),(net[1]-net[5])/scale,3e-10)
   if name in ['FD','LTE']:
    gate('direct FD ratio '+str((T,name)),s['max_log_ratio_residual'],3e-11);gate('entropy sign '+str((T,name)),int(s['entropy']< -1e-27),0)
    if name=='LTE':gate('LTE relative cancellation '+str(T),net[0]/(F[0]+B[0]),3e-11)
    else:gate('direct entropy affinity '+str(T),(s['entropy']-.8*net[0])/(F[0]+B[0]),3e-11)
 phase='author compare '+str(T)
 if T==2.:auth=stored
 else:auth=A.integrate(b,rest_order=28,parent_order=24,orient_order=12,phi_order=16,scenarios=[dict(name='unblocked',mode='unblocked'),dict(name='empty',mode='empty'),dict(name='LTE',mode='FD',mu_nu=eq),dict(name='emission_dominated',mode='FD',mu_nu=eq-.8*T)])
 rows.append(dict(kind='author_reference',T=T,result=auth));save()
 for name in ['unblocked','empty','LTE','FD']:
  aname='emission_dominated'if name=='FD'else name
  for branch,ab in [('F','forward'),('B','backward')]:
   for i,key in enumerate(D.NAMES):
    x=refs[0]['scenarios'][name][branch][i];y=refs[1]['scenarios'][name][branch][i];aa=auth['scenarios'][aname][ab][key]['value']
    if y:
     gate('independent refinement '+str((T,name,branch,key)),x/y-1,3e-6);gate('absolute independent author '+str((T,name,branch,key)),aa/y-1,3e-6)
 save()
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
