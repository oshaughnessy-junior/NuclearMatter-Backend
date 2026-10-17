"""Independent physics-audit probes; imports authors' frozen reference modules."""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'thermodynamics'))
sys.path.insert(0, str(BASE / 'weak_rates'))
import lepton_reference as lr
import static_capture as wr

results = []
def record(name, measured, expected, tolerance):
    error = abs(measured - expected) / max(abs(expected), 1e-100)
    results.append(dict(name=name, measured=float(measured), expected=float(expected),
                        relative_error=float(error), tolerance=tolerance,
                        passed=bool(error < tolerance)))

# Direct momentum integration independently evaluates the kinetic-energy integral.
for T, mu, m in [(2., 3., lr.ELECTRON_MASS), (4., 110., lr.MUON_MASS),
                  (2., -3., lr.ELECTRON_MASS)]:
    st = lr.state_mu(T, mu, m)
    pmax = np.sqrt((abs(mu)+m+100*T)**2-m*m)
    def integ(power, difference):
        def f(p):
            E = np.hypot(p, m)
            a, b = expit((mu-E)/T), expit((-mu-E)/T)
            return p*p*E**power*(a-b if difference else a+b)
        return quad(f, 0, pmax, epsabs=1e-10, epsrel=2e-11)[0]*lr.C
    record(f'momentum density {T,mu,m}', st.net_density, integ(0, True), 2e-9)
    record(f'momentum energy {T,mu,m}', st.energy_density, integ(1, False), 2e-9)

# Check source-level equilibrium, entropy production, and disconnected support.
state = wr.State(T=2., mu_e=3., n_n=7e-6, n_p=3e-6)
munu = state.mu_e-wr.Q+state.T*np.log(state.n_p/state.n_n)
for channel, sign, energies in [('electron_capture',1,[.1,1.,5.,20.]),
                              ('positron_capture',-1,[2.,5.,20.]),
                              ('neutron_decay',-1,[.1,.5,.75])]:
    E = np.array(energies)
    j, chi = wr.spectral(state,E,channel)
    db_error = float(np.max(abs(np.log(j/chi)+(E-sign*munu)/state.T)))
    results.append(dict(name='detailed balance '+channel, absolute_error=db_error,
                        tolerance=2e-13, passed=db_error<2e-13))
    for f in [.01,.2,.8,.99]:
        C=j*(1-f)-chi*f
        entropy=C*(np.log((1-f)/f)-(E-sign*munu)/state.T)
        results.append(dict(name=f'entropy production {channel} f={f}',
                            minimum=float(min(entropy)), passed=bool(min(entropy)>=-1e-13)))

E=np.linspace(wr.Q-wr.ME+1e-5,wr.Q+wr.ME-1e-5,31)
j1,c1=wr.spectral(state,E,'positron_capture')
j2,c2=wr.spectral(state,E,'neutron_decay')
results.append(dict(name='antineutrino disconnected band',
    band_MeV=[wr.Q-wr.ME,wr.Q+wr.ME],
    max_relaxation_s_1=float(np.max(j1+c1+j2+c2)),
    passed=bool(np.all(j1+c1+j2+c2 == 0))))

# Derivative needed to couple total matter energy to evolving composition.
nB, T, Y = 1e-5, 2., .3
st=lr.state_density(T,nB*Y)
uY=-wr.Q+st.chemical_potential-T*st.dmu_dtemperature_fixed_net_density
def u(y):
    return -wr.Q*y+1.5*T+lr.state_density(T,nB*y).energy_density/nB
h=1e-4
record('matter composition energy derivative', (u(Y+h)-u(Y-h))/(2*h),uY,2e-7)
out=dict(scope='Selected independent physics probes, not domain certification',
         checks=results, all_passed=all(r['passed'] for r in results))
(Path(__file__).parent/'probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(checks=len(results),all_passed=out['all_passed']),indent=2))
