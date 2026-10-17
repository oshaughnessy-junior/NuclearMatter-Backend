"""Restricted transparent, isochoric, thermostatted free-nucleon element.

Composes existing research references without modifying them. Evolves mu_e,
with charge neutrality imposed through ideal electron/positron density.
No neutron-star degenerate-nucleon, recoil, in-medium or hydro claims.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
THERMO = HERE.parent / 'thermodynamics' / 'lepton_reference.py'
WEAK = HERE.parent / 'weak_rates' / 'static_capture.py'
sys.dont_write_bytecode = True
sys.path[:0] = [str(THERMO.parent), str(WEAK.parent)]
import lepton_reference as lepton
import static_capture as weak


class Element:
    def __init__(self, T, nb, order=96):
        self.T, self.nb, self.order = float(T), float(nb), order

    def state(self, mu):
        gas = lepton.state_mu(self.T, mu)
        Ye = gas.net_density / self.nb
        if not 0 <= Ye <= 1:
            raise ArithmeticError(f'Composition outside free n,p domain: {Ye}')
        source = weak.source_terms(weak.State(self.T, mu, self.nb*(1-Ye), self.nb*Ye), self.order)
        return gas, Ye, source

    def composition_source(self, mu):
        return self.state(mu)[2]['dYe_dt_s-1']

    def matter_energy(self, gas, Ye):
        # The common proton rest mass is subtracted everywhere as an offset.
        # Neutron mass offset Q, identical classical kinetic 3T/2, exact e+-.
        # Photons are a fixed-T contribution included explicitly but unchanged.
        photon = math.pi**2*self.T**4/(15*lepton.HBARC**3)
        return (1-Ye)*weak.Q + 1.5*self.T + (gas.energy_density+photon)/self.nb

    def rhs(self, t, y):
        gas, Ye, source = self.state(y[0])
        F = source['dYe_dt_s-1']
        dmu = self.nb * F / gas.dn_dmu_fixed_temperature
        nu = source['nu_e_number_fm-3_s-1']/self.nb
        antinu = source['antinu_e_number_fm-3_s-1']/self.nb
        loss = source['neutrino_loss_MeV_fm-3_s-1']/self.nb
        # e_mu = mu*n_mu + T*n_T is a Maxwell/Euler identity.
        u_Y = -weak.Q + y[0] + self.T*gas.dn_dtemperature_fixed_mu/gas.dn_dmu_fixed_temperature
        thermostat = loss + u_Y*F
        return np.array([dmu, nu, antinu, loss, thermostat])

    def root(self):
        upper = lepton.state_density(self.T, .999999999*self.nb).chemical_potential
        return brentq(self.composition_source, 0., upper, xtol=2e-13, rtol=2e-13)


def evolved_case(T, nb):
    model = Element(T, nb)
    root = model.root()
    root_gas, root_Y, root_source = model.state(root)
    gross = (root_source['nu_e_number_fm-3_s-1']+root_source['antinu_e_number_fm-3_s-1'])/nb
    root_residual = abs(root_source['dYe_dt_s-1'])/gross
    assert root_residual < 1e-10
    assert root_source['neutrino_loss_MeV_fm-3_s-1'] > 0
    h = max(abs(root)*2e-4, 1e-5)
    F_mu = (model.composition_source(root+h)-model.composition_source(root-h))/(2*h)
    rate = -F_mu*nb/root_gas.dn_dmu_fixed_temperature
    assert rate > 0
    duration = 20/rate
    # Independent root consistency under weak quadrature refinement.
    refined_root = Element(T, nb, 160).root()
    assert abs(refined_root-root)/max(abs(root), 1e-12) < 1e-8
    # Exact ideal-gas composition derivative checked against state differences.
    gas_hi = lepton.state_mu(T, root+h)
    gas_lo = lepton.state_mu(T, root-h)
    dY = (gas_hi.net_density-gas_lo.net_density)/nb
    fd = (model.matter_energy(gas_hi,gas_hi.net_density/nb)-model.matter_energy(gas_lo,gas_lo.net_density/nb))/dY
    analytic = -weak.Q + root + T*root_gas.dn_dtemperature_fixed_mu/root_gas.dn_dmu_fixed_temperature
    derivative_error = abs(fd-analytic)/max(abs(analytic), 1e-8)
    assert derivative_error < 1e-5
    runs = []
    for initial_Y in (root_Y*.35, root_Y+.65*(1-root_Y)):
        initial_gas = lepton.state_density(T, initial_Y*nb)
        initial_u = model.matter_energy(initial_gas, initial_Y)
        endpoints = []
        for tolerance in (2e-7, 2e-10):
            solution = solve_ivp(model.rhs, (0., duration),
                                 [initial_gas.chemical_potential,0.,0.,0.,0.],
                                 method='DOP853', rtol=tolerance, atol=tolerance*1e-3,
                                 max_step=duration/40, dense_output=True)
            assert solution.success
            sampled = []
            worst_lepton = 0.
            worst_energy = 0.
            for t in np.linspace(0,duration,81):
                y = solution.sol(t)
                gas, Ye, source = model.state(y[0])
                delta_u = model.matter_energy(gas, Ye)-initial_u
                lepton_error = (Ye-initial_Y)+y[1]-y[2]
                energy_error = delta_u - (y[4]-y[3])
                worst_lepton = max(worst_lepton, abs(lepton_error)/max(1.,abs(y[1])+abs(y[2])))
                worst_energy = max(worst_energy, abs(energy_error)/max(abs(delta_u),abs(y[3]),abs(y[4]),1.))
                sampled.append([float(t), float(Ye), float(y[3]), float(y[4])])
            final = solution.y[:,-1]
            final_gas, final_Y, final_source = model.state(final[0])
            assert abs(final_Y-root_Y) < 2e-8
            assert worst_lepton < 2e-7
            assert worst_energy < 2e-7
            # Composition must move monotonically toward the stable root.
            series = np.array(sampled)[:,1]
            assert np.all(np.diff(series)*np.sign(root_Y-initial_Y) >= -1e-10)
            endpoints.append(np.array(sampled))
            runs.append({'initial_Ye': initial_Y, 'rtol': tolerance,
                         'rhs_evaluations': solution.nfev, 'final_Ye': final_Y,
                         'abs_final_Ye_minus_root': abs(final_Y-root_Y),
                         'net_source_at_final_s-1': final_source['dYe_dt_s-1'],
                         'emitted_nue_per_baryon': float(final[1]),
                         'emitted_antinue_per_baryon': float(final[2]),
                         'emitted_energy_MeV_per_baryon': float(final[3]),
                         'external_thermostat_heat_MeV_per_baryon': float(final[4]),
                         'matter_energy_change_MeV_per_baryon': model.matter_energy(final_gas,final_Y)-initial_u,
                         'max_scaled_lepton_ledger_residual': worst_lepton,
                         'max_scaled_energy_ledger_residual': worst_energy,
                         'samples_columns': ['t_s','Ye','escaped_energy_MeV_per_baryon','external_heat_MeV_per_baryon'],
                         'samples': sampled[::8]})
        difference = np.max(np.abs(endpoints[0][:,1]-endpoints[1][:,1]))
        assert difference < 1e-6
    # Estimate domain: classical nucleon phase-space occupancy per spin state.
    wavelength = math.sqrt(2*math.pi*lepton.HBARC**2/(939.*T))
    degeneracy = nb*max(root_Y,1-root_Y)*wavelength**3/2
    return {'T_MeV': T,'nB_fm-3':nb,'root_mu_e_MeV':root,'root_Ye':root_Y,
            'root_net_over_gross_rate':root_residual,
            'root_total_neutrino_loss_MeV_fm-3_s-1':root_source['neutrino_loss_MeV_fm-3_s-1'],
            'root_gross_rate_per_baryon_s-1':gross,
            'kinetic_root_thermodynamic_mu_nue_MeV':root-weak.Q+T*math.log(root_Y/(1-root_Y)),
            'relaxation_rate_s-1':rate,'integration_duration_s':duration,
            'composition_energy_derivative_fd_relative_error':derivative_error,
            'max_classical_nucleon_n_lambda3_per_spin':degeneracy,
            'root_shift_96_to160':abs(root-refined_root),'runs':runs}


def limiting_checks():
    # For a prescribed mu=0 bath the source is exactly affine in Ye.
    T = 2.
    ec = weak.target_moments(T,0.,'electron_capture')[0]
    pc = weak.target_moments(T,0.,'positron_capture')[0]
    nd = weak.target_moments(T,0.,'neutron_decay')[0]
    root = (pc+nd)/(ec+pc+nd)
    rate = ec+pc+nd
    t_end = 4/rate
    initial = .15
    sol = solve_ivp(lambda t,y: [(1-y[0])*(pc+nd)-y[0]*ec],(0,t_end),[initial],rtol=1e-11,atol=1e-13)
    exact = root+(initial-root)*math.exp(-rate*t_end)
    assert abs(sol.y[0,-1]-exact)<1e-10
    dilute = Element(T,1e-13)
    dilute_mu = dilute.root()
    dilute_Y = dilute.state(dilute_mu)[1]
    assert abs(dilute_Y-root)<1e-5
    # Endpoint target counting separately checks the signs of the three events.
    pure_n = weak.source_terms(weak.State(T,0.,1e-7,0.))
    pure_p = weak.source_terms(weak.State(T,0.,0.,1e-7))
    assert pure_n['electron_capture'][0] == 0 and pure_n['dYe_dt_s-1'] > 0
    assert pure_p['positron_capture'][0] == 0 and pure_p['neutron_decay'][0] == 0 and pure_p['dYe_dt_s-1'] < 0
    return {'fixed_mu_analytic_relaxation_abs_error':abs(sol.y[0,-1]-exact),
            'fixed_mu0_analytic_root_Ye':root,
            'selfconsistent_dilute_root_Ye':dilute_Y,
            'dilute_root_abs_difference':abs(dilute_Y-root),
            'pure_target_sign_checks':'PASS'}


def main():
    initial_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (THERMO,WEAK,Path(__file__).resolve())}
    result = {'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
              'command':f'{sys.executable} {Path(__file__).resolve()}',
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'source_sha256':initial_hashes,
              'constants':{'Q_MeV':weak.Q,'electron_mass_MeV':lepton.ELECTRON_MASS},
              'scope':'Transparent fixed-nB fixed-T classical static free-nucleon electron/positron capture and neutron decay; external thermostat required',
              'limits':limiting_checks(),
              'cases':[],
              'limitations':['No recoil, nucleon blocking, interactions, pairing, muons, nuclei, irradiation or trapping',
                             'No dense NS core domain qualification',
                             'Static antineutrino emission/absorption has disconnected support; no global thermal-box claim',
                             'Energy ledger includes explicit external thermostat heat; matter energy alone is not conserved',
                             'Reference quadratures and identities validate restricted equations, not their physical completeness']}
    for T, n in ((.5,1e-8),(2.,1e-7),(5.,1e-5)):
        result['cases'].append(evolved_case(T,n))
        print(f'Completed T={T}, nB={n}', flush=True)
    final_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (THERMO,WEAK,Path(__file__).resolve())}
    if initial_hashes != final_hashes:
        result['status'] = 'REJECTED_DEPENDENCY_CHANGED'
        result['final_sha256'] = final_hashes
        (HERE/'rejected_run.json').write_text(json.dumps(result,indent=2)+'\n')
        raise RuntimeError('A dependency changed during this run; rerun to establish provenance')
    path = HERE/'results.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(path),
                      'root_Ye':[c['root_Ye'] for c in result['cases']],
                      'max_ledger_energy_residual':max(r['max_scaled_energy_ledger_residual'] for c in result['cases'] for r in c['runs'])},indent=2))

if __name__ == '__main__':
    main()
