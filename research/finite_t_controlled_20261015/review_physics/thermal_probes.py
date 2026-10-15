"""Independent targeted coupling checks; no trajectory replay."""
import sys
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'thermal_element'))
import cooling_element as ce

checks=[]
def check(name,error,tolerance):
    checks.append(dict(name=name,scaled_error=float(error),tolerance=tolerance,
                       passed=bool(error<tolerance)))

# Reconstruct the two-variable Jacobian from full returned energy and charge,
# rather than using the author's analytical heat/composition derivatives.
for T,nb,Y in [(2.,1e-7,.3),(5.,1e-5,.5),(1.4,1e-7,.5285271553),
                (3.5,1e-5,.2334007699)]:
    model=ce.CoolingElement(nb)
    gas=ce.lepton.state_density(T,nb*Y)
    mu=gas.chemical_potential
    rhs=model.rhs(0,[T,mu,0,0,0])
    def fields(t,m):
        g=ce.lepton.state_mu(t,m)
        y=g.net_density/nb
        return np.array([model.energy(g,y),y])
    hT,hmu=T*1e-5,max(abs(mu),1)*1e-5
    J=np.column_stack([(fields(T+hT,mu)-fields(T-hT,mu))/(2*hT),
                       (fields(T,mu+hmu)-fields(T,mu-hmu))/(2*hmu)])
    target=np.array([-rhs[4],rhs[3]-rhs[2]])
    independently_solved=np.linalg.solve(J,target)
    err=np.max(abs(independently_solved-rhs[:2])/np.maximum(1.,abs(rhs[:2])))
    check(f'finite-difference two-variable solve {T,nb,Y}',err,2e-7)

def momentum_energy(T,Y,nb):
    g=ce.lepton.state_density(T,Y*nb)
    m=ce.lepton.ELECTRON_MASS
    top=abs(g.chemical_potential)+m+100*T
    pmax=math.sqrt(top*top-m*m)
    def integrand(p):
        E=math.hypot(p,m)
        return p*p*E*(expit((g.chemical_potential-E)/T)+
                       expit((-g.chemical_potential-E)/T))
    e=quad(integrand,0,pmax,epsabs=1e-10,epsrel=2e-12)[0]*ce.lepton.C
    photons=math.pi**2*T**4/(15*ce.lepton.HBARC**3)
    return (1-Y)*ce.weak.Q+1.5*T+(e+photons)/nb

data=json.loads((ROOT/'thermal_element/results.json').read_text())
for run in data['runs']:
    if run['weak_order']!=160: continue
    initial=momentum_energy(run['T0_MeV'],run['Ye0'],run['nB_fm-3'])
    final=momentum_energy(run['final_T_MeV'],run['final_Ye'],run['nB_fm-3'])
    residual=final-initial+run['escaped_energy_MeV_per_baryon']
    check('independent momentum-energy endpoint ledger '+str(run['T0_MeV']),
          abs(residual)/max(1.,abs(initial),abs(run['escaped_energy_MeV_per_baryon'])),1e-8)
    # Conservative composition-independent bound for all T >= the stop floor.
    T=run['Tfloor_MeV']; nb=run['nB_fm-3']
    bound=nb*(2*math.pi*ce.lepton.HBARC**2/(939*T))**1.5/2
    checks.append(dict(name='all-composition classical bound '+str(run['T0_MeV']),
                       upper_bound=bound,passed=bool(bound<.02)))
out=dict(scope='Targeted independent coupling audit, no trajectory replay',
         checks=checks,all_passed=all(c['passed'] for c in checks))
Path(__file__).with_name('thermal_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
