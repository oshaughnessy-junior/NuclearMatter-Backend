"""Independent new-state NR FD EOS checks and factory conventions."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'nr_baryon_eos'))
import free_nucleons as r
checks=[];samples=[]
def check(name,a,b,tol=2e-6,scale=1e-20):
    error=float(np.max(np.abs(np.asarray(a)-b))/max(float(np.max(np.abs(np.asarray(b)))),scale))
    checks.append(dict(name=name,actual=np.asarray(a).tolist(),expected=np.asarray(b).tolist(),relative_error=error,passed=error<tol))
for T,n,O in [(5.,2e-4,1.29333236),(.7,8e-4,100.)]:
    s=r.state_density(T,n,O);samples.append(s.asdict());tag=str((T,n,O))
    emax=max(0.,s.kinetic_mu)+90*T;pmax=math.sqrt(2*r.MASS*emax)
    points=[math.sqrt(2*r.MASS*k) for k in [s.kinetic_mu-30*T,s.kinetic_mu,s.kinetic_mu+30*T] if 0<k<emax]
    density=quad(lambda p:p*p*expit((s.kinetic_mu-p*p/(2*r.MASS))/T),0,pmax,points=points,epsabs=1e-10,epsrel=1e-11)[0]/(math.pi**2*r.HBARC**3)
    kin=quad(lambda p:p**4/(2*r.MASS)*expit((s.kinetic_mu-p*p/(2*r.MASS))/T),0,pmax,points=points,epsabs=1e-10,epsrel=1e-11)[0]/(math.pi**2*r.HBARC**3)
    check(tag+' independent momentum n,K',[s.density,s.kinetic_energy_density],[density,kin],1e-10)
    check(tag+' prescribed density',s.density,n,1e-10)
    check(tag+' covariance positive',float(s.fluctuation_A2*s.fluctuation_A0>s.fluctuation_A1**2),1.,1e-12)
    # Independent coordinate derivatives of the solved canonical state.
    for h in [2e-3,1e-3]:
        tm=r.state_density(T*(1-h),n,O);tp=r.state_density(T*(1+h),n,O)
        nm=r.state_density(T,n*(1-h),O);np_=r.state_density(T,n*(1+h),O)
        check(tag+f' Cv h{h}',(tp.energy_density-tm.energy_density)/(2*h*T),s.cv_fixed_density)
        check(tag+f' muT h{h}',(tp.mu-tm.mu)/(2*h*T),s.dmu_dT_fixed_density)
        check(tag+f' canonical e_n h{h}',(np_.energy_density-nm.energy_density)/(2*h*n),s.mu-T*s.dmu_dT_fixed_density)
        check(tag+f' canonical s_n h{h}',(np_.entropy_density-nm.entropy_density)/(2*h*n),-s.dmu_dT_fixed_density)
        check(tag+f' canonical P_n h{h}',(np_.pressure-nm.pressure)/(2*h*n),n/s.dn_dmu)
        check(tag+f' canonical P_T h{h}',(tp.pressure-tm.pressure)/(2*h*T),s.entropy_density+n*s.dmu_dT_fixed_density)
    refined=r.state_density(T,n,O,rtol=5e-13,tail=90)
    check(tag+' refined moments',[s.pressure,s.entropy_density,s.cv_fixed_density],[refined.pressure,refined.entropy_density,refined.cv_fixed_density],1e-9)

for T,nn,np_ in [(5.,6e-4,2e-4),(.7,.002,.0005)]:
    m=r.matter_state(T,nn,np_);pair=m.pair('neutron');reverse=m.pair('proton')
    recreated=[r.state_mu(T,pair.kinetic_mu_a,pair.offset_a,pair.mass),r.state_mu(T,pair.kinetic_mu_b,pair.offset_b,pair.mass)]
    check(str((T,nn,np_))+' response-factory densities',[s.density for s in recreated],[nn,np_],1e-10)
    check('factory charge neutrality',m.electron.net_density,np_,1e-10)
    check('factory offsets',[pair.offset_a,pair.offset_b],[r.Q,0.],1e-12)
    check('factory reverse fields',[reverse.mu_a,reverse.mu_b],[pair.mu_b,pair.mu_a],1e-12)
    nue=m.optical('nue');anti=m.optical('antinue')
    check('factory chemical equilibrium conventions',[nue.mu_neutrino,anti.mu_neutrino],[m.electron.chemical_potential+m.proton.mu-m.neutron.mu,-m.electron.chemical_potential-m.proton.mu+m.neutron.mu],1e-12)
    shifted=r.matter_state(T,nn,np_,offset_neutron=100+r.Q,offset_proton=100.)
    check('factory common-offset energy',shifted.baryon_energy_density-m.baryon_energy_density,100*(nn+np_),1e-12)
    check('factory common-offset equilibrium invariant',shifted.optical().mu_neutrino,nue.mu_neutrino,1e-12)

out=dict(passed=all(c['passed'] for c in checks),checks=checks,sample_states=samples,source_sha256=hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest())
Path(__file__).with_name('nr_eos_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2))
