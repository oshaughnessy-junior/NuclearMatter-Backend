"""DIFFERENT frozen-candidate review, forward gates precede response checks."""
from pathlib import Path
import json,hashlib,shutil,sys,math,time,importlib.util,dataclasses
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_rmf_port';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.dont_write_bytecode=True;start=time.monotonic();checks=[];states=[];detail={}
assert sha(A/'CANDIDATE_MANIFEST.json')=='185afee22cda5164e62eaff36911125bf7ffc9907427dd552f45cfe322aa55e3';assert sha(A/'MANIFEST.json')=='16dba06db264b9867f196fe5aadc4f04f064bb8323dabad9087487c65ae0fce0'
manifest=json.loads((A/'MANIFEST.json').read_text());before={k:sha(A/k)for k in manifest['files']};assert before==manifest['files'];candidate=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert all(sha(A/'staged'/k)==h for k,h in candidate['files'].items());pure=json.loads((P/'PREPARATION_CHECKPOINT.json').read_text());assert all(sha(P/k)==h for k,h in pure['files'].items())
(P/'INITIAL_AUTHOR_PIN.json').write_text(json.dumps(dict(manifest=sha(A/'MANIFEST.json'),candidate=sha(A/'CANDIDATE_MANIFEST.json'),files=before),indent=2)+'\n')
copy=P/'package_copy';src=A/'test_mirror/src';paths={}
for p in src.rglob('*.py'):
 rel=p.relative_to(src);dest=copy/'src'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);paths[str(rel)]=sha(p);assert sha(dest)==sha(p)
for name in['finite_t_rmf.py','_rmf_moments.py','_rmf_response.py']:
 rel='src/nmbackend/reference/'+name;assert sha(copy/rel)==candidate['files'][rel]
sys.path.insert(0,str(copy/'src'));from nmbackend.reference import finite_t_rmf as M;from nmbackend.reference import _rmf_moments as G
sp=importlib.util.spec_from_file_location('different_review_frozen_momentum_oracle',P/'oracle.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O)
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def save(status='RUNNING'):
 (P/'candidate_results.json').write_text(json.dumps(dict(status=status,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,states=states,details=detail,seconds=time.monotonic()-start),indent=2)+'\n')
def nofail(stage):
 save(stage)
 if any(not c['passed']for c in checks):raise RuntimeError('FAILEDstage '+stage+'; rawpreserved')
ck('actualrealmoduleorigin',0 if Path(M.__file__).is_relative_to(copy)and sha(Path(M.__file__))==candidate['files']['src/nmbackend/reference/finite_t_rmf.py']else 1,0);detail['module_path']=str(M.__file__);detail['package_source_hashes']=paths
L=json.loads((P/'PREPARATION_LOCK.json').read_text());control=json.loads((P/'CONTROL_LOCK.json').read_text());refs=[json.loads((P/f'oracle_state_{i}.json').read_text())for i in[0,1]]
cold=O.state(*control['future_control_cases']['representable_stable_cold_pairunderflow_n_T_Y']);(P/'oracle_cold_boundary.json').write_text(json.dumps(cold,indent=2)+'\n');refs.append(cold)
mp={'energy_density':'energy','pressure':'pressure','entropy_density':'entropy','free_energy_density':'free_energy'};fields={'phi':'phi','omega':'W','rho':'R','dirac_mass':'mD','kinetic_mu_n':'nu_n','kinetic_mu_p':'nu_p','mu_n':'mu_n','mu_p':'mu_p'}
for i,z in enumerate(refs):
 if time.monotonic()-start>105:raise TimeoutError('Forwardreviewbudgetcheckpoint')
 a=M.state_density(z['T'],z['nB'],z['Yp']);states.append(a.asdict())
 for attr,key in mp.items():ck(f'forward{i} {attr}',getattr(a,attr)/z[key]-1,2e-9)
 for attr,key in fields.items():ck(f'forward{i} {attr}',getattr(a,attr)-z[key],2e-7)
 for attr,value in [('neutron_density',z['nB']*(1-z['Yp'])),('proton_density',z['nB']*z['Yp']),('scalar_density',z['neutron']['scalar']+z['proton']['scalar'])]:ck(f'forward{i} '+attr,getattr(a,attr)/value-1,2e-9)
 ck(f'forward{i} rootresidual',a.normalized_equation_residual,2e-9);ck(f'forward{i} canonicalstablebranch',0 if a.locally_admitted and a.scalar_curvature_fm_minus2>1e-8 and a.hessian_min_eigenvalue>0 and a.response_condition<1e8 else 1,0)
 # Separatelogcomponents, no relativebulk qualification of unrepresentedantipairs.
 for sp_,gas in [('n',z['neutron']),('p',z['proton'])]:
  ck(f'forward{i} logantiparticle{sp_}',getattr(a,'log_antiparticle_density_'+sp_)-gas['log_antiparticle'],1e-8)
 nofail('FORWARD'+str(i))
# Only now compare analytic responses to alreadyqualifiedindependentoracles.
for i,(a,z)in enumerate(zip(states,refs)):
 ck(f'response{i} Cv',a['cv_fixed_density_composition']/z['Cv']-1,3e-6);ck(f'response{i} D',a['scalar_curvature_fm_minus2']/z['D']-1,2e-9);HH=np.array(z['H']);ck(f'response{i} fullH',np.linalg.norm(np.array(a['hessian_n_p'])-HH)/np.linalg.norm(HH),2e-6)
rfd=json.loads((P/'pure_fd_results.json').read_text())['Richardson'];a=states[0];HH=np.array(a['hessian_n_p'])
ck('analyticH vsindependentre-solvedRichardsonchemical',np.linalg.norm(HH-np.array(rfd['H']))/np.linalg.norm(HH),2e-6);ck('analyticH vsindependentre-solvedfreeenergysecond',np.linalg.norm(HH-np.array(rfd['H_from_free']))/np.linalg.norm(HH),1e-4);ck('analyticCv vsindependentre-solvedRichardson',a['cv_fixed_density_composition']/rfd['Cv']-1,3e-6)
# Pair-abundant directoriginalmomentum comparison, not just coldlow-anti agreement.
gasmap={'density':'net','scalar':'scalar','energy':'energy','pressure':'pressure','entropy':'entropy','chi':'chi','scalar_nu':'B','scalar_mass':'Q','n_temperature':'n_t','scalar_temperature':'rho_t','s_temperature':'s_t'}
for i,point in enumerate(L['pair_gas_t_nu_m_internal_fm']):
 oracle=json.loads((P/f'pair_gas_{i}.json').read_text());g=G.moments(*point)
 for attr,key in gasmap.items():ck(f'pairrich{i} '+attr,(getattr(g,attr)-oracle[key])/max(abs(oracle[key]),.01),2e-9)
 ck(f'pairrich{i} loganti',g.log_antiparticle_density-oracle['log_antiparticle'],1e-8)
 cv=point[0]*(g.s_temperature-g.n_temperature*g.n_temperature/g.chi);ck(f'pairrich{i} fixedNETCv',cv/oracle['pair_Cv_fixed_net']-1,3e-6)
 nofail('PAIR'+str(i))
# Defaultphysicaladmission vsdiagnosticstationarybranch distinction.
z=json.loads((P/'oracle_state_2.json').read_text());d=M.diagnose_density(z['T'],z['nB'],z['Yp']);detail['spinodal']=d.asdict();ck('spinodal diagnosticnotadmitted',0 if not d.locally_admitted and d.hessian_min_eigenvalue<0 and d.scalar_curvature_fm_minus2>0 else 1,0);ck('spinodaldiagnosticfullH',np.linalg.norm(np.array(d.hessian_n_p)-np.array(z['H']))/np.linalg.norm(z['H']),2e-6)
try:M.state_density(z['T'],z['nB'],z['Yp']);fail=True
except ValueError as e:fail=False;detail['spinodal_default_rejection']=str(e)
ck('defaultrejectsspinodal',1 if fail else 0,0)
ck('coldT1 finiteunrepresentedantipairlogs',0 if states[2]['antiparticle_density_n']is None and states[2]['antiparticle_density_p']is None and all(math.isfinite(states[2][k])for k in['log_antiparticle_density_n','log_antiparticle_density_p','energy_density','cv_fixed_density_composition'])else 1,0)
# StrictAPIadmission. Onlyknowninvalidrequests, no speculativephysicalclassification.
bad=[(float('nan'),.2,.2),(2,float('inf'),.2),(2,.2,float('nan')),(True,.2,.2),(2,np.bool_(True),.2),(2,.2,False),(10**10000,.2,.2),([2],.2,.2),(2,.2,complex(.2,0))]
for T in control['future_control_cases']['invalid_T']:bad.append((T,.213,.237))
for n in control['future_control_cases']['invalid_n']:bad.append((12.3,n,.237))
for Y in control['future_control_cases']['invalid_Y']:bad.append((12.3,.213,Y))
for i,args in enumerate(bad):
 try:M.state_density(*args);ok=False
 except ValueError:ok=True
 ck('strictinvalidphysical'+str(i),0 if ok else 1,0)
for i,kw in enumerate([dict(rtol=0),dict(rtol=True),dict(rtol=float('nan')),dict(tail=29.99999),dict(tail=101),dict(tail=float('nan')),dict(tail=10**10000)]):
 try:M.SolveControls(**kw);ok=False
 except ValueError:ok=True
 ck('strictcontrols'+str(i),0 if ok else 1,0)
for i,seed in enumerate([{},'GM1',dataclasses.replace(M.UniformRMFState(**states[0]),model='differentGM1'),dataclasses.replace(M.UniformRMFState(**states[0]),phi=float('nan')),dataclasses.replace(M.UniformRMFState(**states[0]),kinetic_mu_n=10**10000)]):
 try:M.state_density(12.3,.213,.237,initial=seed);ok=False
 except ValueError:ok=True
 ck('strictseed'+str(i),0 if ok else 1,0)
for i,kw in enumerate([dict(params={'C_sigma':12}),dict(mass_mev=939.565),dict(Lambda_v=.1),dict(controls={})]):
 try:M.state_density(12.3,.213,.237,**kw);ok=False
 except(ValueError,TypeError):ok=True
 ck('rejectunsupportedoverride'+str(i),0 if ok else 1,0)
assert {k:sha(A/k)for k in manifest['files']}==before;ck('allauthorfilesunchanged',0,0);detail['frozen_authorfiles']=before;nofail('PASS');print('PASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
