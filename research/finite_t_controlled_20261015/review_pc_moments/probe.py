from pathlib import Path
import sys,json,time,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
sys.dont_write_bytecode=True
from reference import *
H=Path(__file__).parent;R=H.parent;A=R/'positron_capture_moments';sys.path.insert(0,str(A));import moments as M
initial={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list(A.iterdir())+[R/'relativistic_capture_oracle/events.py',R/'positron_capture/pc_kernel.py',R/'relativistic_crossed_thermal/kernel.py']if p.is_file()};checks=[];rows=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save(phase):
 (H/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,initial_hashes=initial,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
state=json.loads((A/'MATCHED_STATE.json').read_text());b=dict(T=state['T'],mu_n=state['neutron']['chemical_potential'],mu_p=state['proton']['chemical_potential'],mu_e=state['electron']['chemical_potential'])
# Independent full pair density, no use of author's stored species root values.
def density(mu,m):
 upper=math.sqrt((max(m,mu)+80*b['T'])**2-m*m)
 return quad(lambda p:p*p*expit((mu-math.hypot(p,m))/b['T']),0,upper,epsabs=1e-19,epsrel=2e-12)[0]/(math.pi**2*HC**3)
nn=density(b['mu_n'],MN);np_=density(b['mu_p'],MP);em=density(b['mu_e'],ME);ep=density(-b['mu_e'],ME)
for n,e,t in [('n forward density',nn/7e-8-1,3e-10),('p forward density',np_/3e-8-1,3e-10),('actual net charge neutrality',(em-ep)/np_-1,3e-10),('pair density ratio',ep/em/state['electron']['antiparticle_density']*state['electron']['particle_density']-1,3e-10)]:gate(n,e,t)
rows.append(dict(independent_densities=dict(n=nn,p=np_,electron=em,positron=ep,net=em-ep)));save('forward gates complete')
# Fresh literal source-polynomial amplitude vs spinors before total rates.
eb=M.E.Bath(b['T'],MP,MN,b['mu_p'],b['mu_n'],-b['mu_e'],anti=True,ml=ME,Mweak=MW)
for Enu,c in[(3.9,-.61),(11.7,.23),(27.4,.82)]:
 # Rest inverse anti+p event, exact outgoing electron root and boost.
 s=MP*MP+2*MP*Enu;rt=math.sqrt(s);pi=(s-MP*MP)/(2*rt);pf=math.sqrt((s-(MN+ME)**2)*(s-(MN-ME)**2))/(2*rt);lc=np.array([[math.hypot(pf,ME),pf*math.sqrt(1-c*c),0.,pf*c]]);nc=np.array([[math.hypot(pf,MN),-lc[0,1],0.,-lc[0,3]]]);beta=np.array([[0,0,Enu/(MP+Enu)]]);l=boost(lc,beta);n=boost(nc,beta);k=np.array([[Enu,0,0,Enu]]);p=np.array([[MP,0,0,0]])
 actual=amplitude(k,p,l,n)[0];spin=M.E.amplitude(eb,k,p,l,n)[0]/2;gate('independent source polynomial amplitude '+str(Enu),spin/actual-1,5e-12)
for settings in[(20,32,8,45.),(28,44,10,60.),(32,52,12,60.)]:
 ref=integral(b,*settings);rows.append(dict(reference=ref));gate('independent integrated neutron forward '+str(settings),ref['neutron_density']/nn-1,3e-6);gate('independent event shells '+str(settings),ref['max_shell'],3e-13);save('independent total checkpoint '+str(settings))
# Latest author angular-refined completed physical row; no stopped work harvested.
data=json.loads((A/'refinement.json').read_text());print([(r['kind'],r['result'].get('controls'))for r in data['rows']],flush=True)
author=next(r['result']for r in data['rows']if r['result'].get('controls',{}).get('initial_order')==12 and 'scenarios'in r['result'])
last=rows[-1]['reference'];prior=rows[-2]['reference']
for z,name in enumerate(['empty','LTE','noneq']):
 for mode,label in enumerate(['forward','backward']):
  for i,key in enumerate(NAMES):
   target=author['scenarios'][name][label][key]['value']or 0.;v=last['sums'][z][mode][i];coarse=prior['sums'][z][mode][i]
   gate('independent total '+name+' '+label+' '+key,v/target-1 if target else v,3e-6)
   gate('independent own refinement '+name+' '+label+' '+key,coarse/v-1 if v else coarse,3e-6)
 net=np.array(last['net'][z]);scale=last['sums'][z][0][1]+last['sums'][z][1][1]
 gate('independent separate energy ledger '+name,(net[3]-net[4]-net[2]+net[1])/scale,3e-13);gate('independent direct materialloss '+name,(net[5]-net[1])/scale,3e-13)
 if name=='LTE':gate('independent LTE forward/inverse',net[0]/last['sums'][z][0][0],3e-13)
 if name=='noneq':gate('independent entropy source',last['entropy'][z]/author['scenarios'][name]['entropy']-1,3e-6)
rows.append(dict(author_comparison_controls=author['controls']))
for path,sha in initial.items():gate('frozen bytes '+path,float(hashlib.sha256((R/path).read_bytes()).hexdigest()!=sha),0.)
save('complete');print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failed=[c for c in checks if not c['passed']],runtime=time.monotonic()-start)))
