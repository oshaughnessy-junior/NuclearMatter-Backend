"""Independent static lepton-energy quadrature and log-only inventory probes."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'emission_only'
sys.path.insert(0,str(P))
import log_emission as r

EXPECTED='f73fd3504498b696614fb5153362f5d27816e659a8cb5bb43d4467ccbde28207'
assert hashlib.sha256((P/'log_emission.py').read_bytes()).hexdigest()==EXPECTED
payload=json.loads((P/'results.json').read_text())
assert payload['status']=='PASS'
fine=payload['runs'][-1];st=payload['state']
T,mu,nn=st['T_MeV'],st['mu_e_prescribed_MeV'],st['neutron']['density']
checks=[];expected={}
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
pref=r.static.GF**2*r.static.VUD**2*(1+3*r.static.GA**2)/(2*math.pi**3*r.static.HBAR)
for channel in r.CHANNELS:
 vals=[]
 for k in [0,1]:
  def integrand(E):
   pe=math.sqrt((E-r.static.ME)*(E+r.static.ME))
   if channel=='positron_capture':return E*pe*(E+r.static.Q)**(2+k)*expit((-mu-E)/T)
   return E*pe*(r.static.Q-E)**(2+k)*expit((E-mu)/T)
  hi=np.inf if channel=='positron_capture' else r.static.Q
  value=nn*pref*quad(integrand,r.static.ME,hi,epsabs=1e-13,epsrel=2e-12,limit=200)[0]
  vals.append(value)
  measured=math.exp(fine['log_moments'][channel][k])
  add(f'independent static lepton-energy moment {channel} k={k}',abs(measured/value-1),3e-5)
 expected[channel]=vals
for k in [0,1]:
 total=sum(expected[c][k] for c in r.CHANNELS)
 add(f'independent complete named static moment k={k}',abs(math.exp(fine['log_moments']['combined'][k])/total-1),3e-5)

bath=r.make_bath(T,st['neutron']['density'],st['proton']['density'],st['mass_MeV'],mu)
for E in [.22,.9,2.5]:
 logs=r.log_spectra(bath,E)
 add(f'held-out complete channel log-sum E={E}',abs(logs['combined']-np.logaddexp(logs['positron_capture'],logs['neutron_decay'])),1e-10)
 if E==.9:
  flag=r.represented(logs['combined'])
  checks.append(dict(name='held-out gap tails finite but ordinary unavailable',
                     passed=bool(math.isfinite(logs['combined']) and not flag['representable'] and flag['value'] is None)))
for name,sp in [('neutron',bath.neutron),('proton',bath.proton)]:
 z=math.exp(sp.kinetic_mu/T)
 phase=2*(sp.mass*T/(2*math.pi))**1.5/r.static.HBARC**3
 shift=phase*z/sp.density-1
 add(name+' leading FD-to-classical asymptotic density correction',abs(shift-z/2**1.5),2e-11)

failure=ROOT/'quantum_energy_moments/static_inventory_failure.json'
checks.append(dict(name='original strict wrapper failure hash unchanged',
 passed=hashlib.sha256(failure.read_bytes()).hexdigest()==payload['original_failure_sha256']))
assert hashlib.sha256((P/'log_emission.py').read_bytes()).hexdigest()==EXPECTED
out=dict(scope='No heavy integral replay; independent static moments and held-out spectral arithmetic',
 checks=checks,all_passed=all(c['passed'] for c in checks),independent_static_moments=expected,
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'log_emission.py',P/'run_reference.py']})
Path(__file__).with_name('emission_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
