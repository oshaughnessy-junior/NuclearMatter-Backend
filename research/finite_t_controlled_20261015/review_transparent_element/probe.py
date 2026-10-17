"""Fresh-process review; Heun conservative integrator independent of author's Euler."""
import sys,json,hashlib,importlib.util,time,math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];OWN=Path(__file__).parent
sys.dont_write_bytecode=True
checks=[]
def check(name,error,limit):
 checks.append(dict(name=name,error=float(error),limit=float(limit),passed=bool(abs(error)<=limit)))
def hashes():
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['transparent_antinu_element','hot_beta_moments'] for p in (ROOT/d).iterdir() if p.is_file()}
before=hashes();t0=time.monotonic()
spec=importlib.util.spec_from_file_location('transparent_element_reviewed_exact',ROOT/'transparent_antinu_element/element.py');E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
check('clean import no frozen file mutations',float(before!=hashes()),0)
check('no generic study loaded',float('study'in sys.modules),0)
for module,leaf in [(E.H,'hot_beta_moments/hot_moments.py'),(E.K,'relativistic_crossed_thermal/kernel.py'),(E.PC,'review_pc_moments/reference.py')]:check('actual loaded '+leaf,float(Path(module.__file__).resolve()!=ROOT/leaf),0)
old=json.loads((ROOT/'transparent_antinu_element/results.json').read_text());src=E.Sources()
check('transitive vacuum current pinned',float('muon_vacuum/vacuum_cc.py'not in src.pin['sources']),0)
expected='57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88';check('restored beta original results',float(hashlib.sha256((ROOT/'hot_beta_moments/results.json').read_bytes()).hexdigest()!=expected),0)
initial=E.state(2.,.3);z0=src.get(initial)
for name,val,ref in [('initial R',z0['R'],1.1173618686632175e-6),('initial Q',z0['Q'],1.1994635801727575e-5)]:check(name,val/ref-1,1e-8)
# Independent temperature root and actual direct energy sum, no author's inversion.
def invert(u,y):
 def energy(T):
  _,s=E.K.matched_npe(T,E.NB,y)
  return sum(s[k]['energy_density']for k in ('neutron','proton','electron'))/E.NB
 T=brentq(lambda T:energy(T)-u,1.8,2.2,xtol=8e-13)
 return E.state(float(T),float(y))
def rhs(s):
 z=src.get(s);return np.array([-z['Q']/E.NB,z['R']/E.NB]),z
runs=[]
for steps in [2,4,8]:
 s=initial;h=.002/steps;N=Q=0.;rows=[]
 for i in range(steps):
  x=np.array([s['u'],s['Ye']]);a,za=rhs(s);predict=invert(*(x+h*a));b,zb=rhs(predict);xf=x+h*(a+b)/2
  N+=h*(za['R']+zb['R'])/2;Q+=h*(za['Q']+zb['Q'])/2;rows.append(dict(T=s['T'],Ye=s['Ye'],R=za['R'],Q=za['Q']))
  s=invert(*xf)
 check(f'Heun{steps} direct EOS energy ledger',(s['energy_density']+Q-initial['energy_density'])/Q,2e-9)
 check(f'Heun{steps} lepton ledger',(E.NB*(s['Ye']-.3)-N)/N,2e-13)
 e=s['species']['electron'];n=s['species']['neutron'];p=s['species']['proton']
 check(f'Heun{steps} neutrality',(e['particle_density']-e['antiparticle_density'])/(E.NB*s['Ye'])-1,2e-9)
 check(f'Heun{steps} baryon number',(n['density']+p['density'])/E.NB-1,2e-9)
 check(f'Heun{steps} actual positrons positive',float(e['antiparticle_density']<=0),0)
 runs.append(dict(steps=steps,final=s,escaped_number=N,escaped_energy=Q,records=rows))
(OWN/'partial.json').write_text(json.dumps(dict(runs=runs,checks=checks),indent=2))
x=lambda s:np.array([s['u'],s['Ye'],s['T']])
a,b,c=[x(r['final'])for r in runs];ratio=abs((b-c)/(a-b))
for i,k in enumerate(['u','Ye','T']):check('Heun second-order selected '+k,ratio[i]-.25,.035)
euler=[x(r['final'])for r in old['runs']];rich=2*euler[-1]-euler[-2]
for i,k in enumerate(['u','Ye','T']):check('Euler Richardson vs fresh Heun '+k,abs(rich[i]-c[i])/abs(euler[-1][i]-euler[-2][i]),.06)
end=src.get(runs[-1]['final'],fine=True);base=src.get(runs[-1]['final'])
for k in ['R','Q','pc_R','pc_Q','beta_R','beta_Q']:check('heldout Heun endpoint source refinement '+k,base[k]/end[k]-1,1e-7)
check('actual source changes by several percent',float(not .03<1-end['R']/z0['R']<.07),0)
# Finite composition energy partial at independent heldout state.
s=runs[-1]['final'];hy=2e-5;ht=2e-5
uy=(E.state(s['T'],s['Ye']+hy)['u']-E.state(s['T'],s['Ye']-hy)['u'])/(2*hy)
ut=(E.state(s['T']+ht,s['Ye'])['u']-E.state(s['T']-ht,s['Ye'])['u'])/(2*ht)
check('u_Y direct EOS derivative',uy/s['u_Y']-1,2e-6);check('u_T direct EOS derivative',ut/s['u_T']-1,2e-7)
check('composition energy matters',float(abs(s['u_Y']*end['R']/end['Q'])<.04),0)
# Guard source identity BEFORE cached return, synthetic in-memory corruption only.
saved=src.pin;src.pin=dict(saved,numpy='synthetically changed');rejected=False
try:src.get(initial)
except RuntimeError:rejected=True
src.pin=saved;check('stale source identity rejected before cache',float(not rejected),0)
check('all frozen source/raw preserved after probes',float(before!=hashes()),0)
out=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],heun=runs,ratio=ratio.tolist(),euler_richardson=rich.tolist(),endpoint_fine=end,initial_source=z0,identity=src.pin,seconds=time.monotonic()-t0,limitations='Fresh clean-process factory states only; shared reviewed source integrals, different conservative time integrator and EOS root. No external physics uncertainty validation.')
(OWN/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds'],ratio=out['ratio'],heun_final=x(runs[-1]['final']).tolist(),richardson=rich.tolist()),indent=2))
