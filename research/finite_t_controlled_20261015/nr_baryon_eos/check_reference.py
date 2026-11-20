"""NR FD EOS identities, independent phase-space quadrature and limits."""
from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from datetime import datetime,timezone

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit

sys.dont_write_bytecode=True
from free_nucleons import state_mu,state_density,cold_state,matter_state,MASS,HBARC,Q,ROOT

HERE=Path(__file__).resolve().parent
checks=[]
def record(name,error,limit,**context):
    assert math.isfinite(error) and error<limit,(name,error,limit,context)
    checks.append({'name':name,'error':float(error),'limit':limit,**context})


def run():
    for T,n in ((2.,1e-7),(1.,.001),(.05,.005)):
        s=state_density(T,n,Q)
        scale=max(s.kinetic_energy_density,abs(s.energy_density),1e-30)
        record('density_inversion',abs(s.density/n-1),1e-9,T=T,n=n)
        record('Euler',abs(s.energy_density+s.pressure-T*s.entropy_density-s.mu*n)/scale,1e-9,T=T,n=n)
        record('virial',abs(s.pressure/(2*s.kinetic_energy_density/3)-1),1e-14,T=T,n=n)
        record('positive_susceptibility_and_Cv',0. if s.dn_dmu>0 and s.cv_fixed_density>0 else 1.,.5,T=T,n=n)
        # Independent momentum-space scalar quadrature, versus energy-vector path.
        pmax=math.sqrt(2*MASS*(max(s.kinetic_mu,0)+75*T))
        pbreak=math.sqrt(2*MASS*max(s.kinetic_mu,0))
        pref=1/(math.pi**2*HBARC**3)
        nn=quad(lambda p:p*p*expit((s.kinetic_mu-p*p/(2*MASS))/T),0,pmax,points=[pbreak] if pbreak else None,epsabs=1e-9,epsrel=2e-11)[0]*pref
        kk=quad(lambda p:p**4/(2*MASS)*expit((s.kinetic_mu-p*p/(2*MASS))/T),0,pmax,points=[pbreak] if pbreak else None,epsabs=1e-9,epsrel=2e-11)[0]*pref
        record('independent_momentum_density',abs(nn/n-1),1e-9,T=T,n=n)
        record('independent_momentum_energy',abs(kk/s.kinetic_energy_density-1),1e-9,T=T,n=n)
        ss=state_density(T,n,Q,rtol=1e-10,tail=80.)
        for attribute in ('kinetic_mu','kinetic_energy_density','entropy_density','cv_fixed_density','dn_dmu'):
            record('quadrature_refinement_'+attribute,abs(getattr(s,attribute)-getattr(ss,attribute))/max(abs(getattr(s,attribute)),1e-20),1e-8,T=T,n=n)
        for fraction in (4e-4,2e-4,1e-4):
            ht,hn=fraction*T,fraction*n
            hi,lo=state_density(T+ht,n,Q),state_density(T-ht,n,Q)
            cv=(hi.energy_density-lo.energy_density)/(2*ht)
            muT=(hi.mu-lo.mu)/(2*ht)
            record('fixed_density_Cv_FD',abs(cv-s.cv_fixed_density)/s.cv_fixed_density,2e-6,T=T,n=n,step=fraction)
            record('fixed_density_muT_FD',abs(muT-s.dmu_dT_fixed_density)/max(abs(s.dmu_dT_fixed_density),1e-8),2e-6,T=T,n=n,step=fraction)
            hp,lp=state_density(T,n+hn,Q),state_density(T,n-hn,Q)
            muFD=(hp.free_energy_density-lp.free_energy_density)/(2*hn)
            PFD=n*n*((hp.free_energy_density/(n+hn))-(lp.free_energy_density/(n-hn)))/(2*hn)
            record('free_energy_density_derivative',abs(muFD-s.mu)/max(1.,abs(s.mu)),2e-6,T=T,n=n,step=fraction)
            record('pressure_potential_derivative',abs(PFD/s.pressure-1),2e-6,T=T,n=n,step=fraction)
            # Maxwell relation (partial entropy / partial mu)_T = (partial n / partial T)_mu.
            hmu=fraction*max(T,abs(s.kinetic_mu),1.)
            mh,ml=state_mu(T,s.kinetic_mu+hmu,Q),state_mu(T,s.kinetic_mu-hmu,Q)
            maxwell=(mh.entropy_density-ml.entropy_density)/(2*hmu)
            record('Maxwell_entropy_mu',abs(maxwell-s.dn_dT_fixed_mu)/max(abs(s.dn_dT_fixed_mu),1e-20),2e-5,T=T,n=n,step=fraction)
    # Controlled dilute asymptotics with small fugacity.
    T,n=2.,1e-12
    s=state_density(T,n)
    wavelength=math.sqrt(2*math.pi*HBARC**2/(MASS*T))
    nucl=T*math.log(n*wavelength**3/2)
    record('classical_mu',abs(s.kinetic_mu-nucl)/T,1e-7)
    record('classical_pressure',abs(s.pressure/(n*T)-1),1e-7)
    record('classical_energy',abs(s.kinetic_energy_density/(1.5*n*T)-1),1e-7)
    record('classical_Cv',abs(s.cv_fixed_density/(1.5*n)-1),1e-7)
    record('classical_susceptibility',abs(s.dn_dmu/(n/T)-1),1e-7)
    # T/EF <<1 cold and Sommerfeld limits; spin factor fixed independently by pF.
    n=.001; cold=cold_state(n)
    for fraction in (.01,.003):
        T=fraction*cold['EF'];s=state_density(T,n)
        pred=cold['EF']*(1-math.pi**2*fraction*fraction/12)
        record('Sommerfeld_mu',abs(s.kinetic_mu-pred)/cold['EF'],5*fraction**4,T_over_EF=fraction)
        record('cold_energy_approach',abs(s.kinetic_energy_density/cold['kinetic_energy_density']-1),5*fraction*fraction,T_over_EF=fraction)
        record('degenerate_Cv',abs(s.cv_fixed_density/(n*math.pi**2*fraction/2)-1),10*fraction*fraction,T_over_EF=fraction)
    # Common rest-offset shift leaves kinetic physics and all response kinematics unchanged.
    base=state_density(2.,1e-5,0.);shift=state_density(2.,1e-5,100.)
    record('rest_offset_energy',abs((shift.energy_density-base.energy_density)/base.density-100.),1e-10)
    record('rest_offset_mu',abs(shift.mu-base.mu-100.),1e-10)
    record('rest_offset_Cv',abs(shift.cv_fixed_density/base.cv_fixed_density-1),1e-14)
    matter=matter_state(2.,7e-5,3e-5)
    pair=matter.pair();nue=matter.optical();antinue=matter.optical('antinue')
    record('factory_neutrality',abs(matter.electron.net_density/matter.proton.density-1),1e-9)
    record('factory_response_kinetic_mu',abs(pair.kinetic_mu_a-matter.neutron.kinetic_mu),1e-12)
    record('factory_optical_nue_equilibrium',abs(nue.mu_neutrino-(matter.electron.chemical_potential+matter.proton.mu-matter.neutron.mu)),1e-12)
    record('factory_opposite_flavor_mu',abs(nue.mu_neutrino+antinue.mu_neutrino),1e-12)
    diagnostics=[state_density(T,n).asdict() for T,n in ((2.,1e-7),(1.,.001),(.05,.005))]
    return diagnostics,{'T':2.,'nn':7e-5,'np':3e-5,'electron_mu':matter.electron.chemical_potential,
                        'nue_mu':nue.mu_neutrino,'antinue_mu':antinue.mu_neutrino,
                        'neutron':matter.neutron.asdict(),'proton':matter.proton.asdict()}


def main():
    dependencies=[HERE/'free_nucleons.py',Path(__file__).resolve(),ROOT/'thermodynamics'/'lepton_reference.py',
                  ROOT/'response_reference'/'free_density_response.py',ROOT/'allowed_opacity'/'allowed_capture.py',ROOT/'weak_rates'/'static_capture.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    diagnostics,factory=run()
    final={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    result={'status':'PASS' if hashes==final else 'REJECTED_DEPENDENCY_CHANGED','utc':datetime.now(timezone.utc).isoformat(),
            'command':f'{sys.executable} {Path(__file__).resolve()}','source_sha256':hashes,'end_source_sha256':final,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'checks':checks,'checks_passed':len(checks),'sample_species':diagnostics,'factory_sample':factory,
            'scope':'Ideal nonrelativistic FD spin2 common kinetic939MeV nucleons with independent constant rest offsets; exact lepton module reused',
            'publication_status':'Requires independent review; no fluid integration',
            'limitations':['No interactions, antiparticles, pairing, clusters or relativistic dispersion',
                           'T/mass and pF/mass are diagnostics, not a guarantee of omitted-correction accuracy',
                           'Factory matches scalar recoil response/allowed currents only; no claim of crossed optical channel completeness',
                           'Exact vacuum and extreme underflow density are rejected; no log-density interface']}
    (HERE/('results.json' if hashes==final else 'rejected_run.json')).write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='PASS','A dependency changed during this run'
    print(json.dumps({'status':'PASS','checks':len(checks),'diagnostic_pF_over_M':[s['fermi_momentum_over_mass'] for s in diagnostics]},indent=2))

if __name__=='__main__':main()
