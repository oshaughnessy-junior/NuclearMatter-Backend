"""Independent finite-bin DOS, thermodynamic entropy and coupled-step witness."""
from pathlib import Path
import sys,math,time,json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,xlogy
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'thermal_box'))
import box as B
start=time.monotonic();checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def conv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
g=B.grid([0.,3.,9.]);T,Y,n=2.15,.28,1.3e-4
DOS=1/(2*math.pi**2*197.3269804**3)
for k,(a,b)in enumerate(zip(g['edges'][:-1],g['edges'][1:])):
 number=quad(lambda E:DOS*E**2,a,b,epsabs=1e-20)[0]
 energy=quad(lambda E:DOS*E**3,a,b,epsabs=1e-20)[0]
 gate(f'one helicity number bin{k}',g['weights'][k]/number-1,2e-14)
 gate(f'phase weighted energy bin{k}',g['weights'][k]*g['energies'][k]/energy-1,2e-14)

# Actual fresh physical coefficients at a new state and a new selected grid.
r=B.rates(T,Y,n,g,order=48,tail=40.);m=B.eos(T,n,Y);z=m['affinity']
f=np.array([[.25,.07],[.11,.03]])
C=r['j']*(1-f)-r['chi']*f
for a,sigma in enumerate((1.,-1.)):
 for k,E in enumerate(g['energies']):
  gate(f'heldout physicalKMS flavor{a}bin{k}',math.log(r['j'][a,k])-math.log(r['chi'][a,k])+(E-sigma*z)/T,3e-10)

du=-float(np.sum(g['weights']*g['energies']*C))/n
dY=-float(np.sum(np.array([[1.],[-1.]])*g['weights']*C))/n
ds_m=(du-z*dY)/T
ds_r=float(np.sum(g['weights']*np.log((1-f)/f)*C))/n
production=B.entropy_production(T,Y,f,n,g,r)
gate('entropy from material firstlaw and radiation',(ds_m+ds_r)/production-1,3e-11)
gate('nonnegative finite grid entropy source',min(production,0.),0.)

dt=.002/float(np.max(r['j']+r['chi']))
before=B.invariants(T,Y,f,n,g)
newT,newY,newf,exchange=B.step(T,Y,f,n,g,dt,r)
after=B.invariants(newT,newY,newf,n,g)
gate('coupled total energy',after['energy_MeV_per_baryon']/before['energy_MeV_per_baryon']-1,3e-12)
gate('coupled total lepton',after['electron_lepton_per_baryon']-before['electron_lepton_per_baryon'],3e-13)
gate('step signed material energy',exchange['du']/(dt*du)-1,2e-12)
gate('step signed material composition',exchange['dYe']/(dt*dY)-1,2e-12)
gate('actual entropy nondecrease',min(after['entropy_per_baryon']-before['entropy_per_baryon'],0.),0.)
gate('small step approaches entropy tangent',(after['entropy_per_baryon']-before['entropy_per_baryon'])/(dt*production)-1,3e-3)

# Genuine same-grid coefficients change at the new EOS state, not fixed rates.
newr=B.rates(newT,newY,n,g,order=48,tail=40.)
ratechange=float(max(np.max(abs(newr[k]/r[k]-1))for k in ('j','chi')))
gate('new state physical rates change',0. if ratechange>1e-8 else 1.,0.)
try:B.validate_rates(newT,newY,n,g,r)
except ValueError:reject=True
else:reject=False
gate('old state rates rejected',0. if reject else 1.,0.)

# Stationary FD occupations computed from the independently assembled affinity.
f_eq=expit((np.array([[1.],[-1.]])*z-g['energies'])/T)
Ceq=r['j']*(1-f_eq)-r['chi']*f_eq
gate('heldout FD collision stationarity',np.max(abs(Ceq)/(r['j']+r['chi'])),3e-12)
out=dict(passed=all(x['passed']for x in checks),checks=checks,heldout=dict(T=T,Ye=Y,nB=n,grid=g,f=f,rates=r,material_energy_source=du,composition_source=dY,entropy_matter_source=ds_m,entropy_radiation_source=ds_r,entropy_production=production,dt=dt,before=before,after=after,newT=newT,newY=newY,newf=newf,newrates=newr,largest_rate_fractional_change=ratechange),runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'thermal_box/box.py',ROOT/'fd_fluid_element/element.py',ROOT/'quantum_slab/fixed_bath.py']})
(HERE/'results.json').write_text(json.dumps(out,default=conv,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']],rate_change=ratechange)))
if not out['passed']:raise SystemExit(1)
