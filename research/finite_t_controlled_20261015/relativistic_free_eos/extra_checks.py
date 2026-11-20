"""Independent log-antiparticle Bessel witness, empty lepton and thermal limits."""
from free_baryons import *
from scipy.special import kve
import json,time
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
for T,n,mass in [(.001,1e-9,MP),(2.,.002,MN),(40.,.01,MP)]:
 s=state_density(T,n,mass);logMB=math.log(mass*mass*T/(math.pi**2*HBARC**3))+math.log(float(kve(2,mass/T)))-(mass+s.chemical_potential)/T
 gate('independent scaled-Bessel antiparticle log '+str(T),abs(s.log_antiparticle_density-logMB),3e-9)
 rows.append(dict(T=T,n=n,log_antiparticle=s.log_antiparticle_density,independent_Bessel_log=logMB,log_ratio=s.log_antiparticle_to_particle_ratio,ordinary=s.antiparticle_density,representable=s.antiparticle_density_representable))
# Truly empty net electron fraction still retains an e−/e+ plasma.
m=matter_state(3.,.002,0.,.2);gate('zero-net-electron mu zero',abs(m.electron.chemical_potential),1e-20);gate('zero-net-electron particles positive',float(m.electron.particle_density<=0 or m.electron.antiparticle_density<=0),.1);gate('zero-net-electron pair equality',abs(m.electron.particle_density/m.electron.antiparticle_density-1),3e-13)
gate('muon charge closes independent proton density',abs(m.muon.net_density/m.proton.density-1),3e-10)
# Matched cold error ladder should fall by4, consistent with finiteT² corrections.
n=.08;pf=HBARC*(3*math.pi**2*n)**(1/3);ef=math.hypot(MN,pf);z=math.asinh(pf/MN);E0=(pf*ef*(2*pf*pf+MN*MN)-MN**4*z)/(8*math.pi**2*HBARC**3);errors=[]
for T in (.05,.025,.0125):
 s=state_density(T,n);err=s.energy_density-E0;errors.append(err);rows.append(dict(label='cold_thermal_energy',T=T,total_energy=s.energy_density,cold_energy=E0,thermal_difference=err))
for a,b in zip(errors[:-1],errors[1:]):gate('coldT² factor4 '+str(a),abs(a/b/4-1),2e-5)
(P/'extra_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
