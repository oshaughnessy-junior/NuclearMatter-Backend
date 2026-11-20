from pathlib import Path
import sys,json,math,time
from scipy.integrate import quad
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import matter as M
assert json.loads((P/'forward_results.json').read_text())['status']=='PASS'
checks=[];states=[];start=time.monotonic()
def ck(name,a,b,tol):
 err=abs(a-b)/max(abs(a),abs(b),1e-290);checks.append(dict(name=name,a=float(a),b=float(b),relative_error=float(err),gate=tol,passed=bool(err<=tol)))
d=M.LOCK['finite_differences']['relative_step'];tol=M.LOCK['gates']['independent_finite_difference_relative']
for T,n,Y in M.LOCK['selected_forward_baths']:
 for ph in [False,True]:
  s=M.state(T,n,Y,photons=ph);states.append(s.asdict());a=M.state(T*(1+d),n,Y,photons=ph);b=M.state(T*(1-d),n,Y,photons=ph);ck(str((T,n,ph,'e_T')),s.cv,(a.energy-b.energy)/(2*d*T),tol);ck(str((T,n,ph,'s_T')),s.cv/T,(a.entropy-b.entropy)/(2*d*T),tol)
  a=M.state(T,n,Y+d,photons=ph);b=M.state(T,n,Y-d,photons=ph);ck(str((T,n,ph,'e_Y')),s.e_Y,(a.energy-b.energy)/(2*d),tol)
  a=M.state(T,n*(1+d),Y,photons=ph);b=M.state(T,n*(1-d),Y,photons=ph);ck(str((T,n,ph,'e_n')),s.e_n,(a.energy-b.energy)/(2*d*n),tol)
  ck(str((T,n,ph,'Legendre')),s.energy+ s.pressure-T*s.entropy,n*((1-Y)*s.neutron.chemical_potential+Y*(s.proton.chemical_potential+s.electron.chemical_potential)),3e-10)
  ck(str((T,n,ph,'enthalpy photon cancellation')),((s.energy+s.pressure)/n-s.e_n), T*s.entropy/n-T*((1-Y)*s.neutron.dmu_dtemperature_fixed_particle_density+Y*(s.proton.dmu_dtemperature_fixed_particle_density+s.electron.dmu_dtemperature_fixed_net_density)),3e-10)
  if ph:
   q=quad(lambda x:x**3/math.expm1(x),0,100,epsabs=1e-12,epsrel=1e-13)[0]*T**4/(math.pi**2*M.HC**3);ck(str((T,'Planck 2polarizations')),s.photon_energy,q,2e-12);ck(str((T,'photon Cv')),s.photon_cv,4*s.photon_energy/T,2e-12)
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,states=states,seconds=time.monotonic()-start)
(P/'partial_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
if out['failed']:raise SystemExit(1)
