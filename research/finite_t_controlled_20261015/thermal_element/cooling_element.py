"""Isochoric transparent cooling with an explicit shared-mass dilute EOS.

Reference experiment, not dense-matter Urca or radiation hydrodynamics.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
THERMO = HERE.parent/'thermodynamics'/'lepton_reference.py'
WEAK = HERE.parent/'weak_rates'/'static_capture.py'
sys.dont_write_bytecode = True
sys.path[:0] = [str(THERMO.parent),str(WEAK.parent)]
import lepton_reference as lepton
import static_capture as weak

A_GAMMA = math.pi**2/(15*lepton.HBARC**3)
COMMON_NUCLEON_MASS = 939.0


class CoolingElement:
    def __init__(self, nb, *, weak_order=96):
        self.nb, self.weak_order = float(nb), weak_order
        if not math.isfinite(self.nb) or self.nb <= 0:
            raise ValueError('Require positive finite baryon density')

    def energy(self, gas, Ye):
        T = gas.temperature
        # Subtract one common nucleon rest mass per baryon; retain Q once.
        return (1-Ye)*weak.Q + 1.5*T + (gas.energy_density+A_GAMMA*T**4)/self.nb

    def state(self,T,mu):
        if not .20 < T < 10.:
            raise ValueError('RHS outside bounded warm reference domain')
        gas = lepton.state_mu(T,mu)
        Ye = gas.net_density/self.nb
        if not 0 <= Ye <= 1:
            raise ArithmeticError(f'Invalid free-nucleon composition {Ye}')
        wavelength = math.sqrt(2*math.pi*lepton.HBARC**2/(COMMON_NUCLEON_MASS*T))
        degeneracy = self.nb*max(Ye,1-Ye)*wavelength**3/2
        if degeneracy >= .02:
            raise ValueError('Classical nucleon domain exceeded')
        uT = 1.5 + gas.cv_fixed_net_density/self.nb + 4*A_GAMMA*T**3/self.nb
        uY = -weak.Q + mu - T*gas.dmu_dtemperature_fixed_net_density
        if not uT > 0:
            raise ArithmeticError('Nonpositive fixed-composition heat capacity')
        return gas,Ye,uT,uY,degeneracy

    def rhs(self,t,y):
        T,mu = y[:2]
        gas,Ye,uT,uY,_ = self.state(T,mu)
        source = weak.source_terms(weak.State(T,mu,self.nb*(1-Ye),self.nb*Ye),self.weak_order)
        F = source['dYe_dt_s-1']
        loss = source['neutrino_loss_MeV_fm-3_s-1']/self.nb
        Tdot = (-loss-uY*F)/uT
        mudot = (self.nb*F-gas.dn_dtemperature_fixed_mu*Tdot)/gas.dn_dmu_fixed_temperature
        nu = source['nu_e_number_fm-3_s-1']/self.nb
        antinu = source['antinu_e_number_fm-3_s-1']/self.nb
        return np.array([Tdot,mudot,nu,antinu,loss])


def derivative_checks():
    reports = []
    for T,nb,Ye in ((.75,1e-8,.4),(2.,1e-7,.3),(5.,1e-5,.6)):
        model = CoolingElement(nb)
        gas = lepton.state_density(T,Ye*nb)
        _,_,uT,uY,_ = model.state(T,gas.chemical_potential)
        errors = []
        for fraction in (4e-4,2e-4,1e-4):
            hT,hY = fraction*T, fraction
            ehi = model.energy(lepton.state_density(T+hT,nb*Ye),Ye)
            elo = model.energy(lepton.state_density(T-hT,nb*Ye),Ye)
            fdT = (ehi-elo)/(2*hT)
            ehiY = model.energy(lepton.state_density(T,nb*(Ye+hY)),Ye+hY)
            eloY = model.energy(lepton.state_density(T,nb*(Ye-hY)),Ye-hY)
            fdY = (ehiY-eloY)/(2*hY)
            errors.append({'fraction':fraction,'uT_scaled_error':abs(fdT-uT)/max(1.,abs(uT)),
                           'uY_scaled_error':abs(fdY-uY)/max(1.,abs(uY))})
        assert errors[-1]['uT_scaled_error'] < 1e-6
        assert errors[-1]['uY_scaled_error'] < 1e-6
        # Check transformed source produces requested charge and energy derivatives.
        rhs = model.rhs(0,[T,gas.chemical_potential,0,0,0])
        charge_dot = (gas.dn_dtemperature_fixed_mu*rhs[0]+gas.dn_dmu_fixed_temperature*rhs[1])/nb
        assert abs(charge_dot + rhs[2]-rhs[3])/max(1.,rhs[2]+rhs[3]) < 1e-12
        assert abs(uT*rhs[0]+uY*charge_dot+rhs[4])/max(1.,rhs[4]) < 1e-12
        reports.append({'T_MeV':T,'nB_fm-3':nb,'Ye':Ye,'uT_MeV_per_baryon_per_MeV':uT,
                        'uY_MeV_per_baryon':uY,'step_ladder':errors})
    return reports


def run_one(T0,nb,Ye0,rtol,order):
    model = CoolingElement(nb,weak_order=order)
    gas0 = lepton.state_density(T0,Ye0*nb)
    u0 = model.energy(gas0,Ye0)
    Tfloor = .7*T0
    y0 = [T0,gas0.chemical_potential,0.,0.,0.]
    rhs0 = model.rhs(0,y0)
    if rhs0[0] >= 0:
        raise ArithmeticError('Initial case is not thermally cooling')
    initial_timescale = T0/(-rhs0[0])
    def floor(t,y): return y[0]-Tfloor
    floor.terminal, floor.direction = True,-1
    solution = solve_ivp(model.rhs,(0,20*initial_timescale),y0,
                         method='DOP853',rtol=rtol,atol=rtol*1e-3,
                         max_step=initial_timescale/8,events=floor,dense_output=True)
    assert solution.success and len(solution.t_events[0]) == 1
    duration = float(solution.t[-1])
    worst_energy,worst_lepton,max_degeneracy = 0.,0.,0.
    samples=[]
    for t in np.linspace(0,duration,121):
        y = solution.sol(t)
        gas,Ye,uT,uY,degeneracy = model.state(*y[:2])
        u = model.energy(gas,Ye)
        ledgerE = u-u0+y[4]
        ledgerL = Ye-Ye0+y[2]-y[3]
        worst_energy = max(worst_energy,abs(ledgerE)/max(1.,abs(u0),abs(y[4])))
        worst_lepton = max(worst_lepton,abs(ledgerL)/max(1.,abs(y[2])+abs(y[3])))
        max_degeneracy = max(max_degeneracy,degeneracy)
        samples.append([float(t),float(y[0]),float(Ye),float(y[2]),float(y[3]),float(y[4]),float(u)])
    assert worst_energy < 2e-6
    assert worst_lepton < 2e-6
    assert abs(solution.y[0,-1]-Tfloor) < 1e-9
    assert np.all(np.diff(np.array(samples)[:,5]) >= -1e-12)
    assert np.all(np.diff(np.array(samples)[:,6]) <= 1e-8)
    result={'T0_MeV':T0,'nB_fm-3':nb,'Ye0':Ye0,'Tfloor_MeV':Tfloor,'rtol':rtol,
            'weak_order':order,'initial_thermal_cooling_timescale_s':initial_timescale,
            'termination_time_s':duration,'termination':'Prescribed temperature floor',
            'rhs_evaluations':solution.nfev,'initial_u_MeV_per_baryon_common_rest_subtracted':u0,
            'final_T_MeV':float(solution.y[0,-1]),'final_Ye':samples[-1][2],
            'escaped_energy_MeV_per_baryon':float(solution.y[4,-1]),
            'max_scaled_energy_ledger_error':worst_energy,
            'max_scaled_lepton_ledger_error':worst_lepton,
            'max_nucleon_n_lambda3_per_spin':max_degeneracy,
            'samples_columns':['t_s','T_MeV','Ye','escaped_nue_per_baryon','escaped_antinue_per_baryon','escaped_energy_MeV_per_baryon','matter_energy_MeV_per_baryon_common_rest_subtracted'],
            'samples':samples[::10]}
    return result,solution


def main():
    paths=(Path(__file__).resolve(),THERMO,WEAK)
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checks=derivative_checks()
    print('EOS derivative ladder and transformed-source checks PASS',flush=True)
    results=[]
    refinements=[]
    for T,nb,Ye in ((2.,1e-7,.3),(5.,1e-5,.5)):
        group=[]
        for rtol,order in ((2e-6,96),(2e-9,96),(2e-9,160)):
            report,solution=run_one(T,nb,Ye,rtol,order)
            results.append(report); group.append((report,solution))
            print(f'Completed T0={T} nb={nb} rtol={rtol} order={order}',flush=True)
        for index,label in ((0,'ODE tolerance'),(2,'weak quadrature')):
            a,sa=group[index]; b,sb=group[1]
            time_error=abs(a['termination_time_s']/b['termination_time_s']-1)
            times=np.linspace(0,min(sa.t[-1],sb.t[-1]),101)
            ya,yb=sa.sol(times),sb.sol(times)
            scaled_error=float(np.max(np.abs(ya-yb)/np.maximum(1.,np.abs(yb))))
            assert time_error < 1e-5 and scaled_error < 1e-5
            refinements.append({'T0_MeV':T,'nB_fm-3':nb,'kind':label,
                                'relative_event_time_error':time_error,
                                'max_scaled_matched_history_difference':scaled_error})
    final_hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output={'status':'PASS' if hashes==final_hashes else 'REJECTED_DEPENDENCY_CHANGED',
            'utc':datetime.now(timezone.utc).isoformat(),'command':f'{sys.executable} {Path(__file__).resolve()}',
            'source_sha256':hashes,'end_source_sha256':final_hashes,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'constants':{'Q_MeV':weak.Q,'m_e_MeV':lepton.ELECTRON_MASS,'common_nucleon_kinetic_mass_MeV':COMMON_NUCLEON_MASS,'a_gamma_MeV-3_fm-3':A_GAMMA},
            'scope':'No-thermostat isochoric transparent cooling; common-mass classical nucleons, ideal e+- and photons',
            'derivative_checks':checks,'runs':results,'refinement_checks':refinements,
            'publication_status':'Requires independent review before publication',
            'limitations':['No recoil, nucleon blocking, interactions, pairing, nuclei, muons or incident neutrinos',
                           'Photon/e+- pair thermal equilibrium imposed within matter; photons remain local and do not escape',
                           'No hydro, mechanical work, trapping, diffusion or box thermalization',
                           'Static antineutrino support gap preserved; irrelevant to escaped-radiation energy accounting',
                           'Small tested state trajectories only; not production domain qualification']}
    file=HERE/('results.json' if hashes==final_hashes else 'rejected_run.json')
    file.write_text(json.dumps(output,indent=2)+'\n')
    assert output['status']=='PASS','Dependencies changed during execution'
    print(json.dumps({'status':'PASS','runs':len(results),'derivative_states':len(checks),'refinements':len(refinements),
                      'max_energy_ledger':max(r['max_scaled_energy_ledger_error'] for r in results),
                      'max_lepton_ledger':max(r['max_scaled_lepton_ledger_error'] for r in results)},indent=2))

if __name__=='__main__': main()
