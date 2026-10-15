"""Independent endpoint momentum-energy and local FD-derivative recheck."""
from pathlib import Path
import sys,math,json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'fd_fluid_element'
sys.path.insert(0,str(P))
import element as el
manifest=json.loads((P/'CHECKPOINT.json').read_text())
for name in ['element.py','short_evolution.py']:
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==manifest['files_sha256'][name]
data=json.loads((P/'short_evolution_results.json').read_text());n=data['nB']
checks=[]
def add(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
def momentum_u(T,Y):
 m=el.physics.matter_state(T,n*(1-Y),n*Y);energy=0.;pref=1/(math.pi**2*el.physics.HBARC**3)
 for s in [m.neutron,m.proton]:
  end=math.sqrt(2*s.mass*(max(0.,s.kinetic_mu)+80*T))
  value=quad(lambda p:p*p*(s.offset+p*p/(2*s.mass))*expit((s.kinetic_mu-p*p/(2*s.mass))/T),
             0,end,epsabs=1e-10,epsrel=2e-12)[0]
  energy+=pref*value
 e=m.electron;endE=max(abs(e.chemical_potential),e.mass)+80*T
 end=math.sqrt(endE*endE-e.mass*e.mass)
 def integrand(p):
  E=math.hypot(p,e.mass)
  return p*p*E*(expit((e.chemical_potential-E)/T)+expit((-e.chemical_potential-E)/T))
 energy+=pref*quad(integrand,0,end,epsabs=1e-10,epsrel=2e-12)[0]
 return energy/n

u0=momentum_u(2.,.3)
for tag in ['coarse','fine']:
 run=data[tag];end=run['final'];u=momentum_u(end['T'],end['Ye'])
 add(tag+' direct momentum EOS endpoint energy',abs(u/end['u']-1),1e-10)
 add(tag+' direct momentum matter-plus-escaped energy ledger',abs(u+run['escaped_energy_per_baryon']-u0)/max(1.,abs(u0)),1e-10)
 add(tag+' emitted lepton endpoint ledger',abs(end['Ye']+run['escaped_electron_lepton_per_baryon']-.3),1e-13)

T,Y=data['fine']['final']['T'],data['fine']['final']['Ye']
s=el.eos(T,n,Y);hT=T*1e-4;hY=1e-5
fdT=(momentum_u(T+hT,Y)-momentum_u(T-hT,Y))/(2*hT)
fdY=(momentum_u(T,Y+hY)-momentum_u(T,Y-hY))/(2*hY)
add('endpoint genuine FD frozen-composition uT',abs(fdT/s['u_T']-1),2e-7)
add('endpoint genuine FD composition-energy uY',abs(fdY/s['u_Y']-1),2e-7)
rates=data['fresh_physical_source_evaluations']
checks.append(dict(name='no-photon branch selected',passed=data['photons'] is False))
checks.append(dict(name='recorded distinct state sources genuinely change',passed=len({(r['T'],r['Ye']) for r in rates})==len(rates) and len({r['energy_loss_MeV_per_baryon_s'] for r in rates})>1))
for name in ['element.py','short_evolution.py']:
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==manifest['files_sha256'][name]
out=dict(scope='Independent momentum-space EOS endpoint/derivative checks, no source-grid or time integration replay',checks=checks,all_passed=all(c['passed'] for c in checks),
 initial_independent_u_MeV_per_baryon=u0,source_sha256={name:manifest['files_sha256'][name] for name in ['element.py','short_evolution.py']})
Path(__file__).with_name('fd_fluid_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
