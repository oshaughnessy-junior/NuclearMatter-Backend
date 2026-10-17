from pathlib import Path
import json,math,time,hashlib,atexit
import numpy as np
from scipy.special import expit
import box as b
HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];out=dict(phase='initial');cache={};rate_calls=[]
paths=[Path(b.__file__),b.ROOT/'fd_fluid_element/element.py',b.ROOT/'quantum_slab/fixed_bath.py',b.ROOT/'quantum_slab/guarded_capture.py',b.ROOT/'capture_endpoint/mapped_capture.py',b.ROOT/'crossed_opacity/crossed_decay.py',b.ROOT/'nr_baryon_eos/free_nucleons.py',b.ROOT/'thermodynamics/lepton_reference.py',b.ROOT/'response_reference/free_density_response.py']
hashes={str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def convert(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save():
 report=out|dict(checks=checks,passed=out['phase']=='complete'and all(c['passed']for c in checks),runtime_seconds=time.monotonic()-start,physical_rate_evaluations=rate_calls,source_sha256=hashes,end_source_sha256={str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
 (HERE/'results.json').write_text(json.dumps(report,default=convert,indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def deadline():
 if time.monotonic()-start>105:raise TimeoutError('Box105s experiment budget reached; partial checkpoint preserved')
nB=1e-4;T0=2.;Y0=.3;g=b.grid();f0=np.tile(expit(-g['energies']/3.),(2,1));initial=b.invariants(T0,Y0,f0,nB,g)
def physical(T,Y):
 deadline();key=(T,Y)
 if key not in cache:
  cache[key]=b.rates(T,Y,nB,g);rate_calls.append(cache[key])
 return cache[key]
r0=physical(T0,Y0);duration=.08/float(np.max(r0['j']+r0['chi']))
out.update(nB=nB,grid=g,initial=dict(T=T0,Ye=Y0,f=f0,totals=initial),duration_s=duration,method='Explicit Euler df with matching u,Ye update followed by full FD energy inversion',radiation_ansatz='Four energy bins per flavor, bin-constant occupations and one-helicity integrated phase weights; coefficients sampled at phase-weighted energy. No continuum rate claim.')
# Initial physical KMS and nonnegative event entropy-production theorem.
aff=b.eos(T0,nB,Y0)['affinity']
gate('initial physical KMS',np.max(abs(np.log(r0['j']/r0['chi'])+(g['energies']-b.SIGNS*aff)/T0)),3e-10)
gate('initial entropy production positive',float(b.entropy_production(T0,Y0,f0,nB,g,r0)<0),.5)
save();out['phase']='independent_LTE'
# This solve uses total E/Le constraints and FD bin occupations, no evolution.
eq=b.equilibrium(nB,g,initial['energy_MeV_per_baryon'],initial['electron_lepton_per_baryon']);out['LTE']=eq
for k,field in enumerate(['energy_MeV_per_baryon','electron_lepton_per_baryon']):gate('LTE conserved '+field,abs(eq['totals'][field]-initial[field]),3e-10)
gate('LTE entropy exceeds initial',max(0.,initial['entropy_per_baryon']-eq['totals']['entropy_per_baryon']),2e-12)
req=physical(eq['T'],eq['Ye']);Ceq=req['j']*(1-eq['f'])-req['chi']*eq['f']
gate('LTE actual physical collision stationary',np.max(abs(Ceq)/np.maximum(req['j']+req['chi'],1e-300)),3e-12)
save();out['phase']='dynamics';runs=[]
for steps in [2,4,8]:
 T,Y,f=T0,Y0,f0.copy();rows=[];previous=initial
 for k in range(steps):
  r=physical(T,Y);production=b.entropy_production(T,Y,f,nB,g,r);before=dict(T=T,Ye=Y,f=f.copy());T,Y,f,exchange=b.step(T,Y,f,nB,g,duration/steps,r);totals=b.invariants(T,Y,f,nB,g)
  gate(f'{steps}step{k} total energy',abs(totals['energy_MeV_per_baryon']/initial['energy_MeV_per_baryon']-1),3e-12)
  gate(f'{steps}step{k} total electron lepton',abs(totals['electron_lepton_per_baryon']-initial['electron_lepton_per_baryon']),3e-13)
  gate(f'{steps}step{k} finite-step entropy nondecrease',max(0.,previous['entropy_per_baryon']-totals['entropy_per_baryon']),3e-12)
  gate(f'{steps}step{k} interior occupations',float(np.any((f<=0)|(f>=1))),.5)
  gate(f'{steps}step{k} instantaneous entropy production',float(production<0),.5)
  rows.append(dict(before=before,after=dict(T=T,Ye=Y,f=f.copy()),totals=totals,exchange=exchange,entropy_production_per_baryon_s=production,max_dt_relaxation=float(duration/steps*np.max(r['j']+r['chi']))))
  previous=totals
 run=dict(steps=steps,rows=rows,final=dict(T=T,Ye=Y,f=f.copy()),totals=previous);runs.append(run);out['runs']=runs;save()
# Three resolutions check local first-order behavior, not broad trajectory accuracy.
changes={}
for field in ['T','Ye']:
 d24=abs(runs[0]['final'][field]-runs[1]['final'][field]);d48=abs(runs[1]['final'][field]-runs[2]['final'][field]);changes[field]=dict(delta2_4=d24,delta4_8=d48,ratio=d24/d48)
 gate(field+' step refinement decreases',d48/d24,1.)
changes['f_max']=dict(delta2_4=float(np.max(abs(runs[0]['final']['f']-runs[1]['final']['f']))),delta4_8=float(np.max(abs(runs[1]['final']['f']-runs[2]['final']['f']))))
gate('occupation step refinement decreases',changes['f_max']['delta4_8']/changes['f_max']['delta2_4'],1.);out['step_changes']=changes
out['phase']='endpoint_internal_refinement';end=runs[-1]['final'];m=b.eos(end['T'],nB,end['Ye'])['matter'];refinements=[]
for flavor in b.FLAVORS:
 E=float(g['energies'][1]);a=b.material.physics.coefficients(m,flavor,E,order=48,tail=40.);c=b.material.physics.coefficients(m,flavor,E,order=64,tail=50.)
 for key in ['j_s','chi_s']:gate('endpoint selected internal '+flavor+key,abs(a[key]/c[key]-1),3e-6)
 refinements.append(dict(flavor=flavor,E=E,coarse=a,refined=c))
out['endpoint_internal_refinements']=refinements;out['phase']='complete';save();print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),runtime=time.monotonic()-start,duration_s=duration,final=out['runs'][-1]['final'],LTE=dict(T=eq['T'],Ye=eq['Ye']),failed=[c for c in checks if not c['passed']]),default=convert,indent=2))
