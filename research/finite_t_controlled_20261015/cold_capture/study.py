from pathlib import Path
import json,math,time,hashlib
import numpy as np
import resolved as r
HERE=Path(__file__).resolve().parent;start=time.monotonic();rows=[];checks=[];paths=[Path(r.__file__),r.ROOT/'urca_threshold/threshold_study.py',r.ROOT/'urca_threshold/initial_cold_extension_failure.json',r.ROOT/'response_reference/free_density_response.py',r.ROOT/'nr_baryon_eos/free_nucleons.py',r.ROOT/'thermodynamics/lepton_reference.py']
hashes={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def checkpoint():
 report=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
 (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for T in [.01,.03,.007]:
 ref=r.model.cold_reference(.01,.1);matter=r.model.make_state(ref,T);state=matter.optical('nue');E=3*T
 densities=r.model.independent_charge_density(matter);gate(f'T{T} density roots',np.max(abs(np.array(densities)/[.009,.001,.001]-1)),1e-8)
 row=dict(T=T,E=E,reference=ref,mu_e=state.mu_lepton,mu_n=matter.neutron.mu,mu_p=matter.proton.mu,affinity=state.mu_neutrino,independent_densities=densities,methods=[]);rows.append(row);checkpoint()
 methods=[r.adaptive_yq(state,E,order=32,tail=100.),r.adaptive_yq(state,E,order=64,tail=160.),r.gauss_Ec(state,E,order=16,angle_order=24,tail=100.),r.gauss_Ec(state,E,order=24,angle_order=48,tail=160.)]
 row['methods']=methods
 for i,m in enumerate(methods):
  gate(f'T{T} method{i} KMS',abs(m['log_j']-m['log_chi']+(E-state.mu_neutrino)/T),3e-10)
  if 'relative_quad_errors'in m:
   for k,err in enumerate(m['relative_quad_errors']):gate(f'T{T} method{i} adaptive estimated error{k}',err,3e-7)
 for key in ['log_j','log_chi']:
  for i in [0,2,3]:gate(f'T{T} method{i} versus adaptive64 {key}',abs(math.expm1(methods[i][key]-methods[1][key])),3e-6)
 if T==.03:
  core=json.loads((r.ROOT/'urca_threshold/qualified_results.json').read_text())
  old=next(x for x in core['rows']if x['reference']['Yp']==.1 and x['T']==T)
  for key in ['log_j','log_chi']:gate(f'T.03 unchanged qualified core {key}',abs(math.expm1(methods[1][key]-old[key])),3e-6)
 checkpoint()
 print('T',T,'runtime',time.monotonic()-start,'failed',[c for c in checks if not c['passed']],flush=True)
 if time.monotonic()-start>85:raise TimeoutError('Bounded study exceeded85s, partial report preserved')
print('RESULT',len(checks),all(c['passed']for c in checks),time.monotonic()-start)
