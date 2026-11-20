"""Fixed-bath spectral slab using the campaign's static weak ansatz.

No scattering, recoil, nucleon degeneracy, geometry feedback or time evolution.
One neutrino helicity; isotropic material emission, two vacuum boundaries.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import expn

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

thermo = load('slab_lepton_reference', ROOT/'thermodynamics/lepton_reference.py')
weak = load('slab_static_capture', ROOT/'weak_rates/static_capture.py')
LIGHT_CM_S = 2.99792458e10


def energy_grid(temperature, mu, order):
    """Resolve all channel supports; sin^2 mapping regularizes endpoints."""
    upper = max(mu, 0., weak.Q+weak.ME) + 70*temperature
    edges = sorted(set([0., weak.Q-weak.ME, weak.Q+weak.ME, upper] +
                       [x for x in [temperature, 3*temperature, 10*temperature, mu]
                        if 0 < x < upper]))
    nodes, weights = leggauss(order)
    u = .5*(nodes+1)
    energy, w = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        energy.append(a+(b-a)*np.sin(np.pi*u/2)**2)
        w.append(.5*weights*(b-a)*np.pi/2*np.sin(np.pi*u))
    return np.concatenate(energy), np.concatenate(w)


def coefficients(state, energy, flavor):
    channels = ['electron_capture'] if flavor == 'nu_e' else ['positron_capture', 'neutron_decay']
    vals = [weak.spectral(state, energy, channel) for channel in channels]
    return tuple(sum(x[i] for x in vals) for i in range(2))


def escape_integral(tau, angular_order):
    """Integral_0^1 mu*(1-exp(-tau/mu)) dmu, stably at small tau."""
    x, w = leggauss(angular_order)
    mu = .5*(x+1)
    return np.sum(.5*w*mu*(-np.expm1(-np.asarray(tau)[:, None]/mu)), axis=1)


def solve(state, length_cm, energy_order=80, angular_order=128):
    energy, weights = energy_grid(state.T, state.mu_e, energy_order)
    out = {}
    for flavor in ['nu_e', 'anti_nu_e']:
        j, chi = coefficients(state, energy, flavor)
        rate = j+chi
        active = rate > 0
        eq = np.divide(j, rate, out=np.zeros_like(j), where=active)
        tau = rate*length_cm/LIGHT_CM_S
        angular = escape_integral(tau, angular_order)
        dos = weak.DOS_PREFACTOR*energy**2
        # Both outward faces combined. One face carries exactly half.
        flux_spectrum = LIGHT_CM_S*dos*eq*angular
        thin_spectrum = length_cm*dos*j
        out[flavor] = {
            'number_flux_both_faces_cm-2_s-1': float(weights@flux_spectrum*1e39),
            'energy_flux_both_faces_MeV_cm-2_s-1': float(weights@(energy*flux_spectrum)*1e39),
            'thin_number_flux_cm-2_s-1': float(weights@thin_spectrum*1e39),
            'thin_energy_flux_MeV_cm-2_s-1': float(weights@(energy*thin_spectrum)*1e39),
            'energy_escape_fraction': float((weights@(energy*flux_spectrum))/(weights@(energy*thin_spectrum))),
            'occupied_bands': int(np.count_nonzero(active)),
        }
    return out


def main():
    T, nb, ye = 3., 1e-5, .15
    lepton = thermo.state_density(T, nb*ye)
    state = weak.State(T, lepton.chemical_potential, nb*(1-ye), nb*ye)
    j10, chi10 = coefficients(state, np.array([10.]), 'nu_e')
    mean_length = LIGHT_CM_S/float((j10+chi10)[0])
    checks = []
    def check(name, error, bound):
        checks.append({'name': name, 'error': float(error), 'bound': bound, 'pass': bool(error <= bound)})
    def rel(a,b):
        return abs(a-b)/max(abs(a),abs(b),1e-300)
    # Independently known angular escape integral: 1/2-E3(tau).
    for tau in [1e-6, 1e-3, .1, 1., 10., 100.]:
        numerical = escape_integral(np.array([tau]),256)[0]
        expected = .5-expn(3,tau)
        check(f'angular_E3_tau_{tau:g}', rel(numerical,expected), 3e-5)
    # Source-volume vs boundary flux, using independent depth QUADPACK.
    for tau in [.01,.3,3.,30.]:
        def depth_source(z):
            return quad(lambda mu: .5*(np.exp(-tau*z/mu)+np.exp(-tau*(1-z)/mu)),
                        0,1,epsabs=1e-11,epsrel=1e-9)[0]
        volume = tau*quad(depth_source,0,1,epsabs=1e-10,epsrel=1e-8)[0]
        boundary = .5-expn(3,tau)
        check(f'independent_source_boundary_tau_{tau:g}',rel(volume,boundary),2e-7)
    samples = []
    for reference_tau in [1e-5,.1,1.,100.]:
        length=mean_length*reference_tau
        base=solve(state,length)
        refined=solve(state,length,160,256)
        for flavor in base:
            for key in ['number_flux_both_faces_cm-2_s-1','energy_flux_both_faces_MeV_cm-2_s-1']:
                check(f'refinement_{reference_tau:g}_{flavor}_{key}',rel(base[flavor][key],refined[flavor][key]),1e-4)
            f=refined[flavor]['energy_escape_fraction']
            check(f'escape_bounds_{reference_tau:g}_{flavor}', max(0.,-f,f-1),1e-12)
        samples.append({'nu_e_tau_at_10MeV':reference_tau,'length_cm':length,'results':refined})
    source=weak.source_terms(state,192)
    last=samples[-1]
    # Thin spectral integrals must reproduce independently integrated target moments.
    expected_n = {'nu_e': source['nu_e_number_fm-3_s-1'],'anti_nu_e':source['antinu_e_number_fm-3_s-1']}
    for flavor in expected_n:
        got=last['results'][flavor]['thin_number_flux_cm-2_s-1']/last['length_cm']/1e39
        check('spectral_target_number_'+flavor,rel(got,expected_n[flavor]),1e-8)
    summed=sum(last['results'][f]['thin_energy_flux_MeV_cm-2_s-1'] for f in expected_n)/last['length_cm']/1e39
    check('spectral_target_energy',rel(summed,source['neutrino_loss_MeV_fm-3_s-1']),1e-8)
    for flavor in expected_n:
        check('thin_escape_'+flavor,abs(samples[0]['results'][flavor]['energy_escape_fraction']-1),1e-3)
    # The static antineutrino gap is exactly transparent, not equilibrating.
    gap_energy=np.linspace(weak.Q-weak.ME+1e-6,weak.Q+weak.ME-1e-6,31)
    jgap,cgap=coefficients(state,gap_energy,'anti_nu_e')
    check('antineutrino_disconnected_band',float(np.max(np.abs(jgap)+np.abs(cgap))),0.)
    report={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'ansatz':'static classical free nucleons, pair-equilibrated ideal electrons, fixed bath, plane slab, two vacuum boundaries, no scattering',
            'state':{'T_MeV':T,'nB_fm-3':nb,'Ye':ye,'mu_e_total_MeV':state.mu_e},
            'conventions':'j/chi in s^-1; spatial attenuation(j+chi)/c; summed flux of both faces; one helicity',
            'checks':checks,'samples':samples,
            'files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(thermo.__file__),Path(weak.__file__)]}}
    path=Path(__file__).with_name('results.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'failures':[c for c in checks if not c['pass']],'output':str(path)},indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)

if __name__=='__main__':
    main()
