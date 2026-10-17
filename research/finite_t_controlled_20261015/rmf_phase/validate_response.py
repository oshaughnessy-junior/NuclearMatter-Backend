"""Independent finite differences of fixed-nu gas response moments."""
from pathlib import Path
import json
import numpy as np
from stability import *
from gm1_finite_t import gas

checks=[]
for n,T,Y in[(.06,1.,.5),(.16,10.,.1),(.32,30.,.1)]:
 s=state(n,T,Y);tf=T/HBARC;mass=s.dirac_mass_mev/HBARC
 for tag,nu in [('n',s.kinetic_mu_n_mev/HBARC),('p',s.kinetic_mu_p_mev/HBARC)]:
  r=gas_response(tf,nu,mass);h=.001*tf
  plus=gas(tf,nu+h,mass);minus=gas(tf,nu-h,mass)
  pm=gas(tf,nu,mass+h);mm=gas(tf,nu,mass-h)
  vals={'n_nu':(plus['n']-minus['n'])/(2*h),'rho_s_nu':(plus['scalar']-minus['scalar'])/(2*h),'rho_s_m_fixed_nu':(pm['scalar']-mm['scalar'])/(2*h)}
  for key,val in vals.items():
   err=abs(r[key]-val)/max(abs(val),1e-20)
   checks.append(dict(name=f'{n},{T},{Y},{tag} '+key,relative_error=err,passed=bool(err<3e-6)))
  err=abs((pm['n']-mm['n'])/(2*h)+r['rho_s_nu'])/max(abs(r['rho_s_nu']),1e-20)
  checks.append(dict(name=f'{n},{T},{Y},{tag} n_m=-rho_s_nu',relative_error=err,passed=bool(err<3e-6)))
r=dict(passed=all(c['passed'] for c in checks),checks=checks)
Path(__file__).with_name('response_validation.json').write_text(json.dumps(r,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} analytic response FD checks pass")
for c in checks:
 if not c['passed']:print('FAIL',c)
raise SystemExit(0 if r['passed'] else 1)
