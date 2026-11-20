"""Bounded selected-energy optical, angular, normalization and conservation gates."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import time
import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.special import expit,zeta
import scipy
from fixed_bath import *
import static_capture

ROOT=Path(__file__).resolve().parent
start=time.monotonic();checks=[];tables=[]
def check(name,a,b,tol,scale=1e-300):
    error=abs(a-b)/max(abs(b),scale)
    checks.append(dict(name=name,actual=float(a),expected=float(b),scaled_error=float(error),tolerance=tol,passed=bool(error<tol)))

# Absolute one-helicity/one-face blackbody normalization, independent FD integrals.
Tbb=3.
number=LIGHT*DOS/4*quad(lambda E:E*E*expit(-E/Tbb),0.,120*Tbb,epsabs=1e-10,epsrel=1e-11)[0]*1e39
energy=LIGHT*DOS/4*quad(lambda E:E**3*expit(-E/Tbb),0.,120*Tbb,epsabs=1e-9,epsrel=1e-11)[0]*1e39
check('one helicity one face blackbody number',number,3*float(zeta(3))*LIGHT*Tbb**3/(16*math.pi**2*HBARC**3)*1e39,2e-11)
check('one helicity one face blackbody energy',energy,7*math.pi**2*LIGHT*Tbb**4/(960*HBARC**3)*1e39,2e-11)

for tag,T,nn,np_,nueE in [('dilute',2.,7e-5,3e-5,[2.,10.,20.]),('moderate_FD',2.,.002,.001,[20.,40.,60.])]:
    matter=matter_state(T,nn,np_)
    info=dict(tag=tag,T=T,nn=nn,np=np_,mu_e=matter.electron.chemical_potential,
              neutron=matter.neutron.asdict(),proton=matter.proton.asdict())
    for flavor,energies in [('nue',nueE),('antinue',[.4,.85,4.])]:
        for E in energies:
            c=coefficients(matter,flavor,E,order=48)
            refined=coefficients(matter,flavor,E,order=72,tail=50.)
            label=f'{tag}:{flavor}:E={E}'
            # KMS is a separate algebraic check, not proof of integration accuracy.
            check(label+' chemical KMS log',c['log_j']-c['log_chi'],-(E-c['mu_neutrino_MeV'])/T,2e-10,1.)
            for field in ('j_s','chi_s'):
                check(label+' rate refinement '+field,c[field],refined[field],3e-5)
            for name,channel in c['channels'].items():
                check(label+' channel KMS '+name,channel['log_j']-channel['log_chi'],-(E-c['mu_neutrino_MeV'])/T,2e-10,1.)
            expected_eq=float(expit((c['mu_neutrino_MeV']-E)/T))
            check(label+' LTE occupation',c['feq'],expected_eq,3e-11,1e-300)
            # Direct gain/loss cancellation scaled by gross rate, conditioning-aware.
            collision=c['j_s']*(1-expected_eq)-c['chi_s']*expected_eq
            check(label+' LTE direct collision',collision,0.,2e-12,c['rate_s'])
            left,right=boundary_occupations(E)
            physical=fluxes(refined,1e5,left,right)
            transfer=[]
            for tau in (1e-6,.03,1.,30.,1000.):
                length=tau/refined['relaxation_cm_inv']
                flux=fluxes(refined,length,left,right)
                outl,outr=numerical_boundary_flux(refined,tau,left,right)
                A=LIGHT*DOS*E*E*1e39
                check(label+f' tau{tau} angular face left',flux['outgoing_left'],outl,2e-9,A*1e-15)
                check(label+f' tau{tau} angular face right',flux['outgoing_right'],outr,2e-9,A*1e-15)
                volume=numerical_volume_gain(refined,tau,left,right)
                check(label+f' tau{tau} volume conservation',volume,flux['radiation_number_gain'],2e-8,A*1e-15)
                check(label+f' tau{tau} incoming/outgoing conservation',flux['outgoing_both']-flux['incoming_left']-flux['incoming_right'],flux['radiation_number_gain'],3e-8,A*1e-15)
                check(label+f' tau{tau} energy/bath cancellation',flux['radiation_energy_gain_MeV']+flux['bath_energy_source_MeV'],0.,1e-12,max(abs(flux['radiation_energy_gain_MeV']),1.))
                check(label+f' tau{tau} electron lepton cancellation',flux['radiation_electron_lepton_gain']+flux['bath_electron_lepton_source'],0.,1e-12,max(abs(flux['radiation_number_gain']),1.))
                if tau==1e-6:
                    thin=A*tau*(refined['feq']-.5*(left+right))
                    check(label+' thin net flux limit',flux['radiation_number_gain'],thin,2e-5,A*1e-20)
                if tau==1000.:
                    thick=A*refined['feq']/2
                    check(label+' thick two-face outgoing limit',flux['outgoing_both'],thick,1e-10,A*1e-25)
                transfer.append(flux)
            # Independent ray evolution, using direct occupation collision law.
            tau=.3;mu=.4;feq=refined['feq']
            def rhs(t,f):return [(refined['j_s']*(1-f[0])-refined['chi_s']*f[0])/(refined['rate_s']*mu)]
            numerical=solve_ivp(rhs,(0.,tau),[left],rtol=2e-11,atol=1e-13).y[0,-1]
            analytic=ray_occupation(tau,mu,tau,feq,left,right)
            check(label+' direct ray ODE',analytic,numerical,3e-10,1e-9)
            # Identical equilibrium boundary fields give no flux divergence.
            equilibrium=fluxes(refined,1e5,feq,feq)
            check(label+' equilibrium incoming equals outgoing',equilibrium['outgoing_both'],equilibrium['incoming_left']+equilibrium['incoming_right'],2e-12)
            check(label+' equilibrium zero material source',equilibrium['bath_number_source'],0.,1e-12,1.)
            tables.append(dict(state=info,coefficients=refined,coarse_coefficients=c,boundaries=dict(left=left,right=right),fixed_1km_slab=physical,optical_depth_ladder=transfer))

# Mathematical static-target limit: raise the common kinetic mass with fixed
# actual n,p,T and physical offsets. No arbitrary rate multiplier is fitted.
heavy=matter_state(2.,7e-5,3e-5,mass=1e7)
static=static_capture.State(2.,heavy.electron.chemical_potential,7e-5,3e-5)
for flavor,E,channels in [('nue',10.,['electron_capture']),('antinue',.4,['positron_capture','neutron_decay']),('antinue',4.,['positron_capture','neutron_decay'])]:
    dynamic=coefficients(heavy,flavor,E,order=72)
    pairs=[static_capture.spectral(static,E,ch) for ch in channels]
    for i,key in enumerate(('j_s','chi_s')):
        expected=sum(float(v[i]) for v in pairs)
        check(f'heavyM static {flavor} E{E} {key}',dynamic[key],expected,3e-5)

paths=['nr_baryon_eos/free_nucleons.py','capture_endpoint/mapped_capture.py','crossed_opacity/crossed_decay.py','allowed_opacity/allowed_capture.py','response_reference/free_density_response.py','thermodynamics/lepton_reference.py','weak_rates/static_capture.py']
report=dict(passed=all(c['passed'] for c in checks),checks=checks,selected_energy_tables=tables,runtime_seconds=time.monotonic()-start,
            scope='Selected spectral energies only; matched FD free nucleons/leptons, fixed thermal bath, named NR allowed capture+crossed inventory, two irradiated faces; no integrated luminosity',
            source_sha256={p:hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest() for p in paths},
            environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
            qualification='Author numerical checkpoint; mapped-capture and this integration require independent review')
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
for c in checks:
    if not c['passed']:print('FAIL',c)
raise SystemExit(0 if report['passed'] else 1)
