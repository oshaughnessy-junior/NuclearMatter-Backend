"""Independent opaque-band scalar-integral normalization of the physical slab."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

path = Path(__file__).with_name('run_slab_review.py')
spec = importlib.util.spec_from_file_location('slab_review_module', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
T, nb, ye = 3., 1e-5, .15
mu = m.thermo.state_density(T, nb*ye).chemical_potential
state = m.weak.State(T, mu, nb*(1-ye), nb*ye)
munu = mu-m.weak.Q+T*np.log(ye/(1-ye))
j, chi = m.coefficients(state, np.array([10.]), 'nu_e')
length = m.LIGHT_CM_S/float((j+chi)[0])*1e8
out = m.solve(state, length, 160, 256)
checks = []
for flavor in ['nu_e', 'anti_nu_e']:
    nmu = munu if flavor == 'nu_e' else -munu
    bands = ([(0, np.inf)] if flavor == 'nu_e' else
             [(0, m.weak.Q-m.weak.ME), (m.weak.Q+m.weak.ME, np.inf)])
    for power, key in [(2, 'number_flux_both_faces_cm-2_s-1'),
                       (3, 'energy_flux_both_faces_MeV_cm-2_s-1')]:
        expected = .5*m.LIGHT_CM_S*m.weak.DOS_PREFACTOR*1e39*sum(
            quad(lambda E: E**power*expit((nmu-E)/T), a, b,
                 epsabs=1e-9, epsrel=1e-11)[0] for a, b in bands)
        actual = out[flavor][key]
        error = abs(actual/expected-1)
        checks.append(dict(flavor=flavor, moment=power, actual=actual,
                           expected=expected, relative_error=error,
                           passed=error < 1e-7))
Path(__file__).with_name('spectral_slab_blackbody_checks.json').write_text(
    json.dumps(checks, indent=2)+'\n')
print(json.dumps(checks, indent=2))
assert all(row['passed'] for row in checks)
