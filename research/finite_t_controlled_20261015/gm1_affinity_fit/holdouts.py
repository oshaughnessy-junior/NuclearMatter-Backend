"""New blocked physical holdouts AFTER coefficient freeze; no fitting here."""
from pathlib import Path
import sys,json,math,hashlib,time,atexit
import numpy as np
sys.dont_write_bytecode=True
from potential import *
sys.path.insert(0,str(P.parent/'rmf_reference'));import gm1_finite_t as E
sys.path.insert(0,str(P.parent/'rmf_phase'));from stability import implicit_hessian
freeze=json.loads((P/'PRE_HOLDOUT_FREEZE.json').read_text())['files']
def hashes():return {p:hashlib.sha256((P/p).read_bytes()).hexdigest()for p in freeze}
assert hashes()==freeze
points=[]
for n in [.2563,.2598]:
 for Y in [.203,.293]:
  for T in [.12,.77]:points.append(('density',n,T,Y))
for Y in [.3264,.3296]:
 for n in [.1873,.2371]:
  for T in [.12,1.37]:points.append(('composition',n,T,Y))
for n in [.2077,.2279]:
 for Y in [.2133,.2731]:
  for T in [.173,2.63]:points.append(('temperature',n,T,Y))
for T in [.12,.43,2.91]:points.append(('joint',.2597,T,.3297))
points += [('endpoint',.18,.12,.33),('endpoint',.26,3.,.18),('endpoint',.26,.12,.33)]
(P/'HOLDOUT_PROTOCOL.json').write_text(json.dumps(dict(points=points,frozen=freeze,coefficient_edits_forbidden=True,reference_Cv='two decreasing central entropy differences 1e-3T/5e-4T',gates=json.loads((P/'PROTOCOL.json').read_text())['bulk_gates'],balance=.01,root=2e-9,Cv_ladder=2e-6),indent=2)+'\n')
rows=[];checks=[];start=time.monotonic()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save():
 out=dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),complete=len(rows)==len(points),runtime_seconds=time.monotonic()-start,start_frozen=freeze,end_frozen=hashes());(P/'holdout_results.json').write_text(json.dumps(out,indent=2)+'\n')
atexit.register(save)
for group,n,T,Y in points:
 if time.monotonic()-start>110:raise TimeoutError('Bounded heldout EOS experiment')
 s=E.state(n,T,Y);gate('exact root before derivatives',s.max_normalized_equation_residual,2e-9)
 if s.max_normalized_equation_residual>2e-9:raise ValueError('Forward gate failed')
 p=evaluate(n,T,Y);H,meta=implicit_hessian(s);cvs=[]
 for frac in [.001,.0005]:
  h=T*frac;lo=E.state(n,T-h,Y);hi=E.state(n,T+h,Y)
  for state in [lo,hi]:gate('perturbed root',state.max_normalized_equation_residual,2e-9)
  cvs.append(T*(hi.entropy_density-lo.entropy_density)/(2*h))
 gate('reference Cv ladder',abs(cvs[0]/cvs[1]-1),2e-6)
 dm=p['mu_n']-p['mu_p']-s.mu_n_mev+s.mu_p_mev
 err=dict(balance=abs(math.expm1(dm/T)),delta_mu=dm,free=p['free_per_baryon']-s.free_energy_density/n,energy=p['energy_density']/n-s.energy_density/n,pressure=abs(p['pressure']/s.pressure-1),entropy=abs(p['entropy_density']/s.entropy_density-1),Cv=abs(p['cv_fixed_nY']/cvs[-1]-1),Hessian=float(np.linalg.norm(np.array(p['chemical_hessian'])-H)/np.linalg.norm(H)))
 for key,tol in [('balance',.01),('free',.005),('energy',.005),('pressure',.01),('entropy',.01),('Cv',.01),('Hessian',.02)]:gate(group+' '+key,err[key],tol)
 gate('positive fitted Cv',int(p['cv_fixed_nY']<=0),0);gate('positive full H fit+exact',int(min(p['min_chemical_eigenvalue'],np.linalg.eigvalsh(H)[0])<=0),0);gate('canonical scalar curvature',int(meta['canonical_scalar_curvature_fm_minus2']<=0),0)
 rows.append(dict(group=group,n=n,T=T,Y=Y,exact=s.asdict(),prediction=p,reference_cv=cvs,reference_H=H.tolist(),errors=err));save()
for x in [(.2,.119,.2),(.2,3.001,.2),(.179,.5,.2),(.261,.5,.2),(.2,.5,.179),(.2,.5,.331),(math.nan,.5,.2),(.2,math.inf,.2)]:
 try:evaluate(*x);gate('invalid domain reject',1,0)
 except ValueError:gate('invalid domain reject',0,0)
gate('frozen unchanged',int(hashes()!=freeze),0);save();print('RESULT',len(rows),len(checks),all(c['passed']for c in checks),time.monotonic()-start);print('MAX', {k:max(abs(r['errors'][k])for r in rows)for k in rows[0]['errors']});print('FAIL',[c for c in checks if not c['passed']])
