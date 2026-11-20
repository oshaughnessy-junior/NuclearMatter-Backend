"""Independent cold capture: forward densities then adaptive y/direct-cos j,chi."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'cold_capture'));import resolved as r
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
files=['cold_capture/resolved.py','urca_threshold/threshold_study.py','urca_threshold/initial_cold_extension_failure.json','response_reference/free_density_response.py','nr_baryon_eos/free_nucleons.py','thermodynamics/lepton_reference.py']
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files};assert hashes['cold_capture/resolved.py']=='d988860e717ce3a08d8fc2d59a3bdc91c506be80ffb44250e6c55cd67b5354a1'
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def checkpoint(phase):
 (P/'partial.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,source_sha256=hashes,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
def density_nr(sp):
 upper=max(sp.kinetic_mu,0)+100*sp.T
 edges=sorted({0.,upper}|{v for v in [sp.kinetic_mu-k*sp.T for k in (-30,-6,0,6,30)] if 0<v<upper})
 value=sum(quad(lambda e:math.sqrt(e)*expit((sp.kinetic_mu-e)/sp.T),a,b,epsabs=1e-12,epsrel=1e-12)[0] for a,b in zip(edges[:-1],edges[1:]))
 return (2*sp.mass)**1.5/(2*math.pi**2*r.model.eos.HBARC**3)*value

def density_e(el):
 mu=el.chemical_potential;T=el.temperature;m=el.mass;upper=max(abs(mu),m)+100*T
 edges=sorted({m,upper}|{v for v in [mu-k*T for k in (-30,-6,0,6,30)] if m<v<upper})
 value=sum(quad(lambda E:E*math.sqrt((E-m)*(E+m))*(expit((mu-E)/T)-expit((-mu-E)/T)),a,b,epsabs=1e-12,epsrel=1e-12)[0] for a,b in zip(edges[:-1],edges[1:]))
 return value/(math.pi**2*r.model.eos.HBARC**3)

def independent(opt,E,angle_order,tail):
 T=opt.baryons.T;mu=opt.mu_lepton;low=(r.A.ME-mu)/T
 cs,cws=np.polynomial.legendre.leggauss(angle_order)
 # Explicit energy-budget edge omitted from the author's cold-root finder.
 budget=(E+opt.baryons.mu_a-opt.baryons.mu_b-mu)/T
 edges=sorted({low,tail,budget}|{v for v in [-512.,-256.,-128.,-80.,-40.,-20.,-8.,-2.,0.,2.,8.,20.,40.,80.,128.] if low<v<tail})
 edges=[v for v in edges if low<=v<=tail]
 def logs(y):
  if time.monotonic()-start>38:raise TimeoutError('Independent45s numerical bound checkpoint')
  Ee=mu+T*y
  if Ee<=r.A.ME:return [-math.inf,-math.inf]
  p=math.sqrt((Ee-r.A.ME)*(Ee+r.A.ME));terms=[[],[]]
  for c,w in zip(cs,cws):
   q=math.sqrt((E-p)**2+2*E*p*(1-c));common=math.log(r.A.K*T*float(w)*Ee*p*r.A.angular_factor(Ee,float(c)))
   terms[0].append(common+float(log_expit(-y))+r.log_response(opt.baryons.reverse(),q,Ee-E))
   terms[1].append(common+float(log_expit(y))+r.log_response(opt.baryons,q,E-Ee))
  return [float(logsumexp(x)) for x in terms]
 probes=edges[1:-1]+[(a+b)/2 for a,b in zip(edges[:-1],edges[1:])]
 scale=[max(logs(y)[k] for y in probes) for k in (0,1)];values=[];errors=[]
 for k in (0,1):
  value=err=0.
  for a,b in zip(edges[:-1],edges[1:]):
   v,e=quad(lambda y:math.exp(logs(y)[k]-scale[k]),a,b,epsabs=1e-13,epsrel=1e-11,limit=250);value+=v;err+=e
  values.append(scale[k]+math.log(value));errors.append(err/value)
 return dict(logs=values,relative_errors=errors,angle_order=angle_order,tail=tail,explicit_budget_y=budget)
checkpoint('before_forward_or_capture')
for Y,T,E in [(.1,.01,.03),(.102,.009,2.6*.009)]:
 # Per-Y offset reconstructed once and fixed, rather than using vacuum Q.
 nB=.01;M=r.model.eos.MASS;pn=r.model.eos.HBARC*(3*math.pi**2*nB*(1-Y))**(1/3);pp=r.model.eos.HBARC*(3*math.pi**2*nB*Y)**(1/3)
 delta=(pp*pp-pn*pn)/(2*M)+math.hypot(pp,r.A.ME)
 m=r.model.eos.matter_state(T,nB*(1-Y),nB*Y,offset_neutron=delta,offset_proton=0.)
 for name,value,target in [('nn',density_nr(m.neutron),nB*(1-Y)),('np',density_nr(m.proton),nB*Y),('ne_net',density_e(m.electron),nB*Y)]:gate(f'forward density Y{Y}T{T} {name}',abs(value/target-1),2e-9)
 opt=m.optical('nue');row=dict(Y=Y,T=T,E=E,delta=delta,mu_e=opt.mu_lepton,affinity=opt.mu_neutrino);rows.append(row);checkpoint('forward_passed_before_capture')
 independent64=independent(opt,E,64,100.);independent96=independent(opt,E,96,180.)
 author=r.gauss_Ec(opt,E,order=28,angle_order=56,tail=180.)
 row.update(independent64=independent64,independent96=independent96,author_Ecos=author)
 for k,key in enumerate(('log_j','log_chi')):
  gate(f'new direct-cos angular/tail refinement Y{Y} {key}',abs(math.expm1(independent64['logs'][k]-independent96['logs'][k])),3e-7)
  gate(f'root Gauss versus independent adaptive/directcos Y{Y} {key}',abs(math.expm1(author[key]-independent96['logs'][k])),3e-6)
  gate(f'independent adaptive reported error Y{Y} {key}',independent96['relative_errors'][k],3e-7)
 gate(f'independently integrated jchi KMS Y{Y}',abs(independent96['logs'][0]-independent96['logs'][1]+(E-opt.mu_neutrino)/T),3e-10)
 if Y==.1:
  frozen=r.model.mapped_capture.log_coefficients(opt,E,energy_order=72,angle_order=72,tail=60.)
  row['original_mapped72_logs']=frozen;row['old_mapped72_relative_errors']=[math.expm1(frozen[k]-independent96['logs'][k]) for k in (0,1)]
  gate('original mapped72 remains failed same3e-6 gate',float(max(abs(x) for x in row['old_mapped72_relative_errors'])<=3e-6),.1)
 checkpoint('point_complete')
# Fast contracts, no expensive kernels needed for these counterexamples.
def rejects(name,call):
 try:call()
 except (ValueError,ArithmeticError):gate(name,0.,.1)
 else:gate(name,1.,.1)
for name,call in [('NaNenergy',lambda:r.controls(math.nan,24,100)),('zeroenergy',lambda:r.controls(0.,24,100)),('boolorder',lambda:r.controls(.03,True,100)),('floatorder',lambda:r.controls(.03,24.5,100)),('smallorder',lambda:r.controls(.03,7,100)),('lowtail',lambda:r.controls(.03,24,59)),('hightail',lambda:r.controls(.03,24,241)),('NaNtail',lambda:r.controls(.03,24,math.nan)),('nonfiniteLog',lambda:r.represent(math.inf))]:rejects('contract '+name,call)
nondegenerate=r.model.a.OpticalState(opt.baryons,0.)
rejects('contract adaptive rejects nondegenerate electron domain',lambda:r.adaptive_yq(nondegenerate,.03))
rejects('contract gauss rejects nondegenerate electron domain',lambda:r.gauss_Ec(nondegenerate,.03))
rejects('contract adaptive rejects NaNrtol',lambda:r.adaptive_yq(opt,.03,rtol=math.nan))
rejects('contract adaptive rejects too-smallrtol',lambda:r.adaptive_yq(opt,.03,rtol=1e-14))
rejects('contract adaptive rejects collapsed q interval',lambda:r.adaptive_yq(opt,1e-30))
author_pin=json.loads((ROOT/'cold_capture/CHECKPOINT.json').read_text())['source_sha256']
gate('all six hashes match declared author freeze',float(any(author_pin[k]!=v for k,v in hashes.items())),.1)
for value in (-1000.,-55933.,710.):
 rep=r.represent(value);gate(f'unrepresentable log{value} retained and None',float(not(rep['log']==value and rep['value'] is None and not rep['representable'])),.1)
rep=r.represent(-745.);gate('rounded positive IEEEsubnormal stays positive',float(not(rep['representable'] and rep['value']>0)),.1)
final={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files};gate('all original model/failure/source hashes unchanged',float(hashes!=final),.1)
report=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,rows=rows,source_sha256=hashes,end_source_sha256=final,runtime_seconds=time.monotonic()-start,scope='Two selected cold directjchi numerical checks; shared physicalresponse/current, no globalcoldcertificate; originalmapped72 failure retained')
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
