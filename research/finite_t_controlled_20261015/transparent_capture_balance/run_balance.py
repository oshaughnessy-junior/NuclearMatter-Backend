"""Bounded balance study. Importing this file never executes scientific sources."""
from pathlib import Path
import importlib.util,sys,json,time,math
import numpy as np
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('transparent_balance_unique',HERE/'balance.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
def run():
 acceptance=M.require_acceptance();start=time.monotonic();rows=[];checks=[]
 def check(n,error,tol):
  c=dict(name=n,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol));checks.append(c)
  if not c['passed']:raise ArithmeticError('Preserved gate failure: '+n)
 def save():
  (HERE/'partial_results.json').write_text(json.dumps(dict(rows=rows,checks=checks,seconds=time.monotonic()-start),indent=2)+'\n')
 def source(y):
  if time.monotonic()-start>110:raise TimeoutError('Bounded110s study; completed states preserved')
  coarse=M.evaluate(float(y));fine=M.evaluate(float(y),True);rows.append(dict(coarse=coarse,refined=fine));save()
  for label,r in [('coarse',coarse),('refined',fine)]:
   for key,val in r['root_residuals'].items():check(f'EOS {y} {label} {key}',val,1e-10)
   check(f'EOS netcharge {y} {label}',r['state']['electron']['net_density']/(1e-7*y)-1,1e-10)
   for key,val in r['density_integral_residuals'].items():check(f'FD integrated density {y} {label} {key}',val,3e-8)
   check(f'energy {y} {label}',r['energy_ledger'],3e-10);check(f'directenergy {y} {label}',r['direct_energy_ledger'],3e-10)
   check(f'Le {y} {label}',r['lepton_ledger']/sum(r['rates'].values()),1e-14)
   if not r['escaping_energy']>0:raise ArithmeticError('Positive actual escaping energy required')
  for ch in fine['channels']:
   for k,v in fine['channels'][ch].items():
    if k in coarse['channels'][ch]:check(f'quadrature {y} {ch} {k}',coarse['channels'][ch][k]/v-1,3e-6)
  save();return fine['net_material_number_source']
 pre=M.preflight()
 if not all(c['passed']for c in pre['checks']):raise ArithmeticError('Forward EOS/current preflight failed')
 chemical=pre['chemical_balance_Ye']
 # Original accepted source point first, with actual kernels freshly executed.
 source(.3);pilot=rows[-1]['refined'];check('originalPCnumber',pilot['rates']['PC']/1.117319281268152e-6-1,3e-6);check('originalECnumber',pilot['rates']['EC']/3.30886034824735e-7-1,3e-6);check('originalbeta',pilot['rates']['beta']/4.2587395065654314e-11-1,3e-6);save()
 # Declared bracket; no assumed coincidence with chemical equilibrium.
 lo=.3;hi=.8
 root=brentq(source,lo,hi,xtol=2e-9,rtol=1e-12)
 source(root);balanced=rows[-1]['refined'];check('kinetic root normalizeddrift',balanced['net_material_number_source']/sum(balanced['rates'].values()),1e-7)
 source(chemical);chemrow=rows[-1]['refined']
 result=dict(accepted_EC_review=acceptance,preflight=pre,rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],kinetic_balance_Ye=root,chemical_balance_Ye=chemical,balanced=balanced,chemical_source=chemrow,seconds=time.monotonic()-start,scope='Selected transparent free-npe beta+PC+EC event inventory; prescribed T2,nB1e-7, empty radiation; no coolingODE or allinteraction equilibrium')
 (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['kinetic_balance_Ye','chemical_balance_Ye','passed','failed','seconds']}))
if __name__=='__main__':run()
