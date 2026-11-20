"""Independent selected cold support/occupation and held-out quadrature audit."""
from pathlib import Path
import sys,math,json,hashlib,time
import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit
ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'cold_capture'))
import resolved as R
from free_density_response import PairState,log_response,cold_response

start=time.monotonic();checks=[];examples=[]
def gate(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and abs(error)<=tol)))

def direct_log(state,q,omega):
 # Integrate transverse momentum squared expressed as initial kinetic energy.
 M,T=state.mass,state.T;w=omega-state.offset_b+state.offset_a
 pz=(2*M*w-q*q)/(2*q);emin=pz*pz/(2*M)
 na,nb=state.mu_a-state.offset_a,state.mu_b-state.offset_b
 upper=max(emin,na,nb-w)+100*T
 edges=sorted({emin,upper}|{x for edge in (na,nb-w) for k in (-40,-8,0,8,40) if emin<(x:=edge+k*T)<upper})
 def lf(t):return float(log_expit((na-t)/T)+log_expit((t+w-nb)/T))
 scale=max(lf(t) for t in edges)
 val=sum(quad(lambda t:math.exp(lf(t)-scale),a,b,epsabs=2e-14,epsrel=3e-12,limit=200)[0] for a,b in zip(edges[:-1],edges[1:]))
 return 2*math.log(M)-math.log(math.pi*q)-3*math.log(R.model.eos.HBARC)+scale+math.log(val)

ref=R.model.cold_reference(.01,.1)
state=R.model.make_state(ref,.01).optical('nue');M=state.baryons.mass
for E,c in [(ref['mu_e_cold']-.2,-.9),(ref['mu_e_cold'],0.),(ref['mu_e_cold']+.2,.9)]:
 Enu=.03;p=math.sqrt(E*E-R.A.ME**2);q=math.sqrt(Enu*Enu+p*p-2*Enu*p*c);om=Enu-E
 for s,omx,label in [(state.baryons,om,'np'),(state.baryons.reverse(),-om,'pn')]:
  independent=direct_log(s,q,omx);actual=log_response(s,q,omx)
  gate(f'FDintegral {label} Ee{E} cos{c}',math.expm1(actual-independent),3e-10)

# Cold geometric interval from perpendicular disks, independent of helper.
cold=PairState(1e-6,M,ref['offset_neutron'],0.,ref['offset_neutron']+ref['EF_n'],ref['EF_p'])
for Enu in (.03,.32,.9):
 for frac,c in [(0.,-.95),(.5,-.95),(1.,.8)]:
  E=ref['mu_e_cold']+frac*Enu;p=math.sqrt(E*E-R.A.ME**2)
  q=math.sqrt(Enu*Enu+p*p-2*Enu*p*c);omega=Enu-E;w=omega+cold.offset_a
  pz=M*w/q-q/2
  initial_disk=max(0.,ref['pFn']**2-pz*pz)
  blocked_disk=max(0.,ref['pFp']**2-(pz+q)**2)
  radial_area=max(0.,initial_disk-blocked_disk) if abs(pz)<ref['pFn'] else 0.
  expected=M*radial_area/(2*math.pi*q*R.model.eos.HBARC**3)
  actual=cold_response(M,cold.offset_a,0.,cold.mu_a,cold.mu_b,q,omega)
  gate(f'cold disks E{Enu} frac{frac} c{c}',(actual-expected)/max(expected,1e-12),2e-11)
  # Away from either edge finite-T response approaches the cold interval.
  if expected>1e-10:
   gate(f'coldlimit E{Enu} frac{frac}',math.expm1(log_response(cold,q,omega)-math.log(expected)),3e-8)
  examples.append(dict(Enu=Enu,Ee=E,cosine=c,pz=pz,initial_disk=initial_disk,blocked_disk=blocked_disk,cold_response=expected))

# New selected spectral point, not among author's three E=3T points.
Enu=.045
a=R.adaptive_yq(state,Enu,order=72,tail=160.)
b=R.gauss_Ec(state,Enu,order=32,angle_order=64,tail=180.)
for key in ('log_j','log_chi'):
 gate('heldout coordinate '+key,math.expm1(a[key]-b[key]),3e-6)
gate('heldout independent coefficient KMS',a['log_j']-a['log_chi']+(Enu-state.mu_neutrino)/.01,3e-10)
old=R.model.guarded_capture.log_coefficients(state,.03,energy_order=72,angle_order=72,tail=60.)
new=R.gauss_Ec(state,.03,order=28,angle_order=56,tail=160.)
old_errors={key:math.expm1(old[k]-new[key]) for k,key in enumerate(('log_j','log_chi'))}
for key,error in old_errors.items():gate('original72 still fails unchanged3e-6 '+key,0. if abs(error)>3e-6 else 1.,0.)
out=dict(passed=all(x['passed']for x in checks),checks=checks,cold_geometry=examples,heldout=dict(T=.01,Enu=Enu,adaptive=a,gauss=b),original_failure_replay=dict(old_logs=list(old[:2]),resolved=new,old_fractional_errors=old_errors),runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'cold_capture/resolved.py',ROOT/'response_reference/free_density_response.py',ROOT/'urca_threshold/initial_cold_extension_failure.json']})
Path(__file__).with_name('cold_capture_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],seconds=out['runtime_seconds'])))
if not out['passed']:raise SystemExit(1)
