"""Frozen-model heldout and signed-moment gates; NO retuning."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from scipy.optimize import brentq
P=Path(__file__).parent;sys.dont_write_bytecode=True
if (P/'validation_results.json').exists():raise RuntimeError('Refusing to overwrite frozen validation; preserve/copy into a fresh owned leaf first')
def load(n,f):
 sp=importlib.util.spec_from_file_location(n,P/f);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
A=load('locked_moment_fit_api','api.py');R=load('validated_physical_fit_refs','reference.py');D=load('independent_fit_phase_reference','phase_reference.py');checks=[];start=time.monotonic()
def ck(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def dump(n,o):(P/n).write_text(json.dumps(o,indent=2))
reg=json.loads((P/'PREREGISTRATION.json').read_text());lock=json.loads((P/'FITTING_LOCK.json').read_text());held=json.loads((P/'heldout_references.json').read_text());train=json.loads((P/'training_references.json').read_text())
if len(held['rows'])!=6 or held['failed']:raise ValueError('Complete qualified preregistered heldout references required')
ck('locked before heldout existed',float(not lock['heldout_source_file_absent']),0);ck('unchanged locked model coefficients',float(hashlib.sha256((P/'model.json').read_bytes()).hexdigest()!=lock['model_sha']),0)
rows=[]
for part,data in [('training',train),('heldout',held)]:
 for row in data['rows']:
  E=row['E'];fit=A.evaluate(E);ref=row['fine']['total'];err={}
  for key in reg['positive_observables']:
   e=fit[key]/ref[key]-1;ck(part+' positivefit '+str((E,key)),e,.01);err[key]=e
  e=(fit['neutrino_energy_change']-ref['neutrino_energy_change'])/(ref['rate']*2.);ck(part+' signedRT '+str(E),e,.03);err['signed_RT']=e
  if abs(ref['neutrino_energy_change'])/(ref['rate']*2.)>=.2:
   rel=fit['neutrino_energy_change']/ref['neutrino_energy_change']-1;ck(part+' signedrelativeaway '+str(E),rel,.05);err['signed_relative_away']=rel
  rows.append(dict(part=part,E=E,fit=fit,reference=ref,errors=err))
dump('validation_partial.json',dict(checks=checks,rows=rows,model_sha=lock['model_sha']))
# Preregistered different-coordinate actual forward anchor, both targets.
coarse=D.integrated(7.,12,16,100.,70.);fine=D.integrated(7.,18,24,150.,90.);ref=next(z['fine']['total']for z in held['rows']if z['E']==7.)
for key in reg['positive_observables']:
 ck('independentphase quadrature '+key,coarse['total'][key]/fine['total'][key]-1,1e-4);ck('independentphase vsCM '+key,fine['total'][key]/ref[key]-1,1e-4)
ck('independentphase signedRT',(fine['total']['neutrino_energy_change']-ref['neutrino_energy_change'])/(ref['rate']*2),1e-4)
# Additional predeclared algorithm, data-dependent ZERO diagnostics, no new fit.
energies=np.geomspace(1.,30.,100);sgn=[A.evaluate(float(e))['neutrino_energy_change']for e in energies];roots=[]
for lo,hi,fl,fh in zip(energies[:-1],energies[1:],sgn[:-1],sgn[1:]):
 if fl*fh<0:roots.append(brentq(lambda e:A.evaluate(float(e))['neutrino_energy_change'],lo,hi))
zero=[]
for root in roots:
 for E in[root*.99,root,root*1.01]:
  detailed=R.detailed(float(E));fit=A.evaluate(float(E));error=(fit['neutrino_energy_change']-detailed['total']['neutrino_energy_change'])/(detailed['total']['rate']*2);ck('postlock zero diagnostic RT '+str(E),error,.03);zero.append(dict(E=E,fit=fit,reference=detailed,error_RT=error))
# Explicit fallback: actual quadrature call compared with saved fresh detailed row.
fallback=A.evaluate(7.,method='detailed')
for key in reg['positive_observables']:ck('explicit actual fallback '+key,fallback[key]/ref[key]-1,2e-12)
for name,kw in [('lowE',dict(E=.9)),('highE',dict(E=31.)),('NaNE',dict(E=float('nan'))),('boolE',dict(E=True)),('wrongT',dict(E=7.,T=2.01)),('wrongmu',dict(E=7.,mu_e=0.)),('flavor',dict(E=7.,flavor='anti_nu_e')),('FDblocked',dict(E=7.,outgoing='FD')),('targets',dict(E=7.,targets=('e-',))),('method',dict(E=7.,method='surrogate'))]:
 try:A.evaluate(**kw);accepted=True
 except ValueError:accepted=False
 ck('reject unsupported '+name,float(accepted),0)
# Input coefficient hash admission is tested in memory, no artifact mutation.
saved=A.MODEL_SHA;A.MODEL_SHA='0'*64
try:A.evaluate(7.);accepted=True
except RuntimeError:accepted=False
A.MODEL_SHA=saved;ck('reject changed coefficient identity',float(accepted),0)
ck('coefficients still exactly locked',float(hashlib.sha256((P/'model.json').read_bytes()).hexdigest()!=lock['model_sha']),0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,independentphase=dict(coarse=coarse,fine=fine,CM_reference=ref),zero_diagnostics=zero,explicit_fallback=fallback,model_sha=lock['model_sha'],seconds=time.monotonic()-start)
dump('validation_results.json',out);print(json.dumps(dict(passed=out['passed'],failed=out['failed'],roots=roots,seconds=out['seconds']),indent=2))
if __name__=='__main__':pass
