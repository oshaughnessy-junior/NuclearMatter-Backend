"""New post-fit sign diagnostic, explicitly outside qualified signed-fit contract."""
from pathlib import Path
import ast,math,json,sys,importlib.util
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
import time
HERE=Path(__file__).resolve().parent;A=HERE.parent/'lepton_scattering_fits';sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('fresh_fit_sign_api',A/'api.py');API=importlib.util.module_from_spec(sp);sys.modules[sp.name]=API;sp.loader.exec_module(API)
E=8.14 # Predeclared additional diagnostic before source evaluation, not a new holdout.
fit=API.evaluate(E);actual=API.evaluate(E,method='detailed')
(HERE/'sign_partial.json').write_text(json.dumps(dict(E=E,fit=fit,actual=actual),indent=2)+'\n')
src=ast.parse((HERE/'phase_probe.py').read_bytes());fns=[n for n in src.body if isinstance(n,ast.FunctionDef)and n.name in ['D','integrate']]
env=dict(math=math,time=time,np=np,quad=quad,expit=expit,leggauss=leggauss,T=2.,MU=.17446333200826397,M=.51099895069,GF=1.1663785e-11,HB=6.582119569e-22)
exec(compile(ast.Module(body=fns,type_ignores=[]),str(HERE/'phase_probe.py'),'exec'),env)
a=env['integrate'](E,18,24,100.,65.);b=env['integrate'](E,26,36,140.,90.)
checks=[]
for key in ['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']:
 scale=2*actual['rate']if key=='neutrino_energy_change'else actual[key]
 for label,x,y in [('independent source refinement',a['moments'][key],b['moments'][key]),('independent source vs CM',b['moments'][key],actual[key])]:
  error=(x-y)/scale;checks.append(dict(name=label+' '+key,error=error,tolerance=1e-4,passed=abs(error)<=1e-4))
 if key!='neutrino_energy_change':
  error=fit[key]/actual[key]-1;checks.append(dict(name='positive fit '+key,error=error,tolerance=.01,passed=abs(error)<=.01))
signedRT=(fit['neutrino_energy_change']-actual['neutrino_energy_change'])/(2*actual['rate'])
checks.append(dict(name='unchanged signed absoluteRT diagnostic',error=signedRT,tolerance=.03,passed=abs(signedRT)<=.03))
checks.append(dict(name='global signed-fit rejection flag remainsFalse',passed=fit['signed_energy_RT_gate_pass']is False))
diag=dict(name='NEW postfit diagnostic signed heating/cooling sign match',passed=bool(fit['neutrino_energy_change']*actual['neutrino_energy_change']>0),fit=fit['neutrino_energy_change'],actual_CM=actual['neutrino_energy_change'],independent_phase=b['moments']['neutrino_energy_change'],classification='Excluded signed-output quality; not a failure of accepted positive moments or an admitted signedfit')
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],sign_diagnostic=diag,E=E,fit=fit,actual_CM=actual,independent_coarse=a,independent_fine=b,scope='Adaptive postfit diagnostic outside qualified signedfit; no refit or claim of preregisteredholdout')
(HERE/'sign_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed'],sign=diag)))
