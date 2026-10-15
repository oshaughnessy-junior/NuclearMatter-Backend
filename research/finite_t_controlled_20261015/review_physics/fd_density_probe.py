"""Held-out entropy/work witness: root bracket and Simpson log-density work."""
from pathlib import Path
import sys,time,math,json,hashlib
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'fd_prescribed_density'))
import density_element as D
start=time.monotonic();D.set_budget(28.);checks=[];calls=[]
def state(T,n,Y,photons=True):
 s=D.state(T,n,Y,photons=photons)
 calls.append(dict(T=T,nB=n,Ye=Y,photons=photons,u=s['u_MeV_per_baryon'],S=s['entropy_per_baryon'],P=s['P_MeV_fm3']))
 return s
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and abs(error)<=tol)))

T0,n0,Y=2.4,1.3e-4,.27
initial=state(T0,n0,Y);S=initial['entropy_per_baryon'];extent=.003
points=[]
for x in (0.,extent/2,extent):
 n=n0*math.exp(x)
 T=T0 if x==0 else brentq(lambda T:state(T,n,Y)['entropy_per_baryon']-S,2.39,2.42,xtol=2e-13,rtol=2e-13)
 s=initial if x==0 else state(T,n,Y)
 points.append(dict(x=x,n=n,T=T,u=s['u_MeV_per_baryon'],S=s['entropy_per_baryon'],P=s['P_MeV_fm3']))
 gate(f'independent bracket entropy x{x}',s['entropy_per_baryon']/S-1,3e-11)
work=extent/6*(points[0]['P']/points[0]['n']+4*points[1]['P']/points[1]['n']+points[2]['P']/points[2]['n'])
energy=points[-1]['u']-points[0]['u']
gate('heldout Simpson pressure work vs endpoint energy',(work-energy)/energy,2e-9)
gate('positive compression energy',0. if work>0 and energy>0 else 1.,0.)
gate('compression temperature increases',0. if points[-1]['T']>T0 else 1.,0.)

no=state(T0,n0,Y,False);eg=math.pi**2*T0**4/(15*D.element.physics.HBARC**3)
gate('photon density derivative',(initial['u_n_MeV_fm3']-no['u_n_MeV_fm3'])/(-eg/n0**2)-1,3e-11)
gate('photon heat derivative',(initial['u_T']-no['u_T'])/(4*eg/(T0*n0))-1,3e-11)
gate('photon composition derivative',initial['u_Y']-no['u_Y'],2e-12)

# A common constant nucleon rest offset cancels from the density derivative.
m=no['matter'];n,p,e=m.neutron,m.proton,m.electron
An=n.mu-T0*n.dmu_dT_fixed_density;Ap=p.mu-T0*p.dmu_dT_fixed_density;Ae=e.chemical_potential-T0*e.dmu_dtemperature_fixed_net_density
shift=37.
shifted=((1-Y)*(An+shift)+Y*(Ap+shift+Ae)-(no['u_MeV_per_baryon']+shift))/n0
gate('common rest zero density derivative',(shifted-no['u_n_MeV_fm3'])/no['u_n_MeV_fm3'],3e-12)
out=dict(passed=all(x['passed']for x in checks),checks=checks,heldout=dict(T0=T0,n0=n0,Ye=Y,photons=True,log_compression=extent,points=points,pressure_work=work,energy_difference=energy),eos_calls=calls,runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'fd_prescribed_density/density_element.py',ROOT/'fd_fluid_element/element.py',ROOT/'nr_baryon_eos/free_nucleons.py',ROOT/'thermodynamics/lepton_reference.py']})
Path(__file__).with_name('fd_density_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],work=work,deltaU=energy,Tend=points[-1]['T'],failed=[x for x in checks if not x['passed']])) )
if not out['passed']:raise SystemExit(1)
