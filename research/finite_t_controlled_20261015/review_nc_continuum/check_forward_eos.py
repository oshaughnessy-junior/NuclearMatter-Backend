from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
p=Path(__file__).resolve().parent;root=p.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(root/'nc_continuum_grid'));import redistribution as r
start=time.monotonic();rows=[];checks=[]
for T,n in [(1.1,.0017),(.2,.0042),(.08,.0042)]:
 s=r.nc.state_density(T,n,0.);mass=s.mass;nu=s.kinetic_mu
 # Independent momentum integral, not the EOS kinetic-energy-coordinate integral.
 cuts=sorted(set([0.,math.sqrt(2*mass*(max(nu,0)+70*T))]+[math.sqrt(2*mass*x) for x in (nu-12*T,nu-4*T,nu,nu+4*T,nu+12*T) if x>0]))
 val=sum(quad(lambda momentum:momentum**2*expit((nu-momentum**2/(2*mass))/T),a,b,epsabs=1e-10,epsrel=2e-12)[0] for a,b in zip(cuts[:-1],cuts[1:]))/(math.pi**2*r.nc.HBARC**3)
 rows.append(dict(T=T,target_density=n,EOS_density=float(s.density),independent_momentum_density=val,chemical_potential=nu,pF_over_M=s.fermi_momentum_over_mass,T_over_M=s.temperature_over_mass))
 for label,error in [('EOS inverse target',abs(s.density/n-1)),('independent momentum target',abs(val/n-1))]:checks.append(dict(name=f'T{T} '+label,error=error,tolerance=3e-11,passed=bool(error<3e-11)))
(p/'forward_eos.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
