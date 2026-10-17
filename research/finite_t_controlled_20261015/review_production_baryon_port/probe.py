"""Different independent particle-only forward, response and domain audit."""
from pathlib import Path
import sys,json,hashlib,importlib.util,math,inspect,shutil
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_baryon_port';sys.dont_write_bytecode=True
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();SOURCE='f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b';MANIFEST='199716b691d88a4fc9b83ed7f8e8b48e36fbff16fcc9b1e8684d18d0922fc523'
assert sha(A/'CANDIDATE_MANIFEST.json')==MANIFEST;assert sha(A/'staged/src/nmbackend/reference/finite_t_baryons.py')==SOURCE
before={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_baryons as B
from nmbackend.constants import M_N,M_P,HBARC
assert sha(Path(B.__file__))==SOURCE
sp=importlib.util.spec_from_file_location('production_baryon_independent_oracle',P/'oracle.py');O=importlib.util.module_from_spec(sp);sys.modules[sp.name]=O;sp.loader.exec_module(O)
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());oracle=json.loads((P/'oracle_results.json').read_text());checks=[];rows=[];derivatives=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
MAP={'number':'particle_density','kinetic':'kinetic_energy_density','energy':'energy_density','pressure':'pressure','entropy':'entropy_density','free_energy':'free_energy_density','chi':'dn_dmu_fixed_temperature','n_T':'dn_dtemperature_fixed_mu','mu_T':'dmu_dtemperature_fixed_particle_density','cv':'cv_fixed_particle_density'}
ck('exactNMBneutronmass',M_N-O.MN,0);ck('exactNMBprotonmass',M_P-O.MP,0);ck('exactNMBhc',HBARC-O.HC,0)
for i,(point,ref)in enumerate(zip(lock['states'],oracle['rows'])):
 s=B.state_density(point['T'],point['n'],point['mass'])if point['kind']=='density'else B.state_mu(point['T'],point['mu'],point['mass'])
 row=dict(index=i,state=s.asdict(),oracle=ref);rows.append(row)
 for k,name in MAP.items():ck('independentmomentum '+str((i,k)),getattr(s,name)/ref[k]-1,lock['forward_gate']if k in['number','kinetic','energy','pressure','entropy','free_energy']else lock['response_gate'])
 if point['kind']=='density':ck('positiveparticle root '+str(i),s.particle_density/point['n']-1,1e-9)
 ck('restkineticexactidentity '+str(i),(s.energy_density-s.mass*s.particle_density-s.kinetic_energy_density)/s.energy_density,3e-12)
 ck('Legendreidentity '+str(i),(s.energy_density-s.temperature*s.entropy_density-s.free_energy_density)/s.energy_density,3e-12)
 ck('positivecenteredCv '+str(i),0 if s.cv_fixed_particle_density>0 else 1,0)
 (P/'partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
# Explicit original capture masses: never conflate them with NMB defaults.
C=importlib.util.spec_from_file_location('review_original_capture_eos_source',R/'hot_electron_capture_moments/electron_capture.py');E=importlib.util.module_from_spec(C);sys.modules[C.name]=E;C.loader.exec_module(E)
for mass,target,key in[(939.56542052,7e-8,'neutron'),(938.27208816,3e-8,'proton')]:
 s=B.state_density(2.,target,mass);_,old=E.matched(2.,1e-7,.3);ck('explicitcapturemassmu '+key,s.chemical_potential-old[key]['chemical_potential'],3e-10);ck('explicitcapturemasskinetic '+key,s.kinetic_energy_density/old[key]['kinetic_energy_density']-1,3e-10)
# Do not proceed to derivative claims on any failed unperturbed forward model.
if any(not c['passed']for c in checks):
 (P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,failed=[c for c in checks if not c['passed']],status='FORWARD_BLOCKED'),indent=2)+'\n');raise ArithmeticError('Forward failure; derivative work not admitted')
# New forward dilute negativeTOTALmu and verytiny represented positiveparticle number.
for T,mu in[(20.,-38.),(1.,M_N-47.3)]:
 s=B.state_mu(T,mu);r=O.momentum(T,mu,M_N)
 for k in['number','kinetic','energy','pressure','entropy','cv']:ck('freshparticleonly signedmu '+str((T,mu,k)),getattr(s,MAP[k])/r[k]-1,3e-10)
for n in[1e-300,1e-310]:
 s=B.state_density(1.,n);ck('tinypositiveparticle root '+str(n),s.particle_density/n-1,1e-9);ck('tinypositiveCv '+str(n),0 if s.cv_fixed_particle_density>0 else 1,0)
# Re-solved density and fixedmu ladders: Richardson removes known centralO(h²).
for idx in[0,2,5]:
 p=lock['states'][idx];s=B.state_density(p['T'],p['n'],p['mass']);T=s.temperature;mu=s.chemical_potential;m=s.mass;n=s.particle_density;est=[]
 for frac in[.001,.0005]:
  h=T*frac;lo=B.state_density(T-h,n,m);hi=B.state_density(T+h,n,m);lm=B.state_mu(T-h,mu,m);hp=B.state_mu(T+h,mu,m);ml=B.state_mu(T,mu-h,m);mh=B.state_mu(T,mu+h,m)
  est.append(dict(cv=(hi.kinetic_energy_density-lo.kinetic_energy_density)/(2*h),muT=(hi.chemical_potential-lo.chemical_potential)/(2*h),P_T=(hp.pressure-lm.pressure)/(2*h),n_T=(hp.particle_density-lm.particle_density)/(2*h),P_mu=(mh.pressure-ml.pressure)/(2*h),chi=(mh.particle_density-ml.particle_density)/(2*h)))
 extrap={k:(4*est[1][k]-est[0][k])/3 for k in est[0]};targets=dict(cv=s.cv_fixed_particle_density,muT=s.dmu_dtemperature_fixed_particle_density,P_T=s.entropy_density,n_T=s.dn_dtemperature_fixed_mu,P_mu=n,chi=s.dn_dmu_fixed_temperature)
 for k in extrap:ck('freshFDladder '+str((idx,k)),extrap[k]/targets[k]-1,2e-6)
 derivatives.append(dict(index=idx,ladder=est,richardson=extrap,targets=targets))
# Cold relativistic specificheat andNR dilute limit: approximationgates explicit.
for idx in[3,4]:
 s=B.state_density(lock['states'][idx]['T'],lock['states'][idx]['n'],lock['states'][idx]['mass']);pf=HBARC*(3*math.pi**2*s.particle_density)**(1/3);ef=math.hypot(pf,s.mass);cv=s.particle_density*math.pi**2*ef*s.temperature/(pf*pf);ck('coldSommerfeldleadingCv '+str(idx),s.cv_fixed_particle_density/cv-1,2e-5)
s=B.state_mu(.23,M_N-31.7*.23);ck('NRdilute Cvperparticle',s.cv_fixed_particle_density/(1.5*s.particle_density)-1,.002)
# API/domain controls distinguish nonphysicalnumbers from omittedpairs.
for name,fn,args,kw in[
 ('zeroT',B.state_mu,(0.,1.),{}),('negativeT',B.state_mu,(-1.,1.),{}),('boolT',B.state_mu,(True,1.),{}),('nanmu',B.state_mu,(1.,float('nan')),{}),('infmass',B.state_mu,(1.,1.,float('inf')),{}),('zeromass',B.state_mu,(1.,1.,0.),{}),('boolmu',B.state_mu,(1.,False),{}),('arraymu',B.state_mu,(1.,np.array(1.)),{}),('badrtol',B.state_mu,(1.,1.),dict(rtol=1e-4)),('boolrtol',B.state_mu,(1.,1.),dict(rtol=True)),('badtail',B.state_mu,(1.,1.),dict(tail=29.)),('booltail',B.state_mu,(1.,1.),dict(tail=False)),('zeroDensity',B.state_density,(1.,0.),{}),('negativeDensity',B.state_density,(1.,-1e-5),{}),('boolDensity',B.state_density,(1.,True),{}),('nanDensity',B.state_density,(1.,float('nan')),{}),('arrayDensity',B.state_density,(1.,np.array(.1)),{})]:
 try:fn(*args,**kw);rejected=False
 except ValueError:rejected=True
 ck('rejectinvalid '+name,0 if rejected else 1,0)
for name,fn,args in[('unrepresentedpopulation',B.state_mu,(1.,M_N-1000)),('unrepresentedinterval',B.state_mu,(1e-300,1e308)),('unrepresentedroot',B.state_density,(1e-300,1e308))]:
 try:fn(*args);rejected=False
 except ArithmeticError:rejected=True
 ck('rejectunrepresented '+name,0 if rejected else 1,0)
for f,h in json.loads((A/'CANDIDATE_MANIFEST.json').read_text())['files'].items():ck('candidatefivepath '+f,0 if sha(A/'staged'/f)==h else 1,0)
after={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};ck('wholeauthorunchanged',0 if before==after else 1,0)
out=dict(status='ACCEPTED_SELECTED_REFERENCE'if all(c['passed']for c in checks)else'BLOCKED',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,derivatives=derivatives,source_sha256=SOURCE,manifest_sha256=MANIFEST,author_unchanged=before==after)
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
