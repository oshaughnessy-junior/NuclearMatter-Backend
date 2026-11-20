"""Blocked physical holdouts, derivative accuracy and two-species stability."""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from numpy.polynomial import chebyshev as ch
from potential import *
ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'rmf_reference'));sys.path.insert(0,str(ROOT/'rmf_phase'))
import gm1_finite_t as exact
from stability import implicit_hessian
start=time.monotonic();checks=[];training=[];rows=[];failures=[]
paths=['rmf_reference/gm1_finite_t.py','rmf_phase/stability.py']
hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def checkpoint(phase):
 (P/'partial.json').write_text(json.dumps(dict(phase=phase,training=training,rows=rows,checks=checks,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
def budget():
 if time.monotonic()-start>105:checkpoint('budget_stop');raise TimeoutError('GM1 fit physical numerical budget reached')
# All density/composition fit samples are at or below .24/.30; blocked
# density/composition slices and joint corners lie beyond those sample hulls.
for n in [.18,.195,.21,.225,.24]:
 for Y in [.18,.22,.26,.30]:
  budget();cold=exact.cold_state(n,Y);mass=cold.dirac_mass_mev/exact.HBARC
  gamma=sum(exact.kf(z*n)*math.hypot(exact.kf(z*n),mass) for z in (1-Y,Y))/(3*exact.HBARC)
  a2=-gamma/(2*n);warm=exact.state(n,2.,Y)
  a4=(warm.free_energy_density/n-cold.free_energy_density/n-a2*4)/16
  training.append(dict(n=n,Y=Y,cold=cold.asdict(),warmT2=warm.asdict(),cold_field=cold.free_energy_density/n-M,T2_field=a2,T4_field=a4))
  gate(f'train primal residual n{n}Y{Y}',warm.max_normalized_equation_residual,2e-8);checkpoint('training')

def fit(degrees):
 V=ch.chebvander2d([scale(x['n'],DOMAIN['n']) for x in training],[scale(x['Y'],DOMAIN['Y']) for x in training],degrees).reshape(len(training),-1)
 fields={k:np.linalg.lstsq(V,np.array([x[k+'_field'] for x in training]),rcond=None)[0].reshape(degrees[0]+1,degrees[1]+1).tolist() for k in ['cold','T2','T4']}
 return fields,float(np.linalg.cond(V))
# Deliberately predeclared smaller model is retained whether it passes or fails.
baseline,cond0=fit((1,1));coeff,cond=fit((4,3))
(P/'coefficients.json').write_text(json.dumps(dict(fields=coeff,domain=DOMAIN,degrees=[4,3],parameters=60,training_states=40,condition_number=cond,form='a=M+C0(n,Y)+T²C2(n,Y)+T⁴C4(n,Y)'),indent=2)+'\n')
(P/'baseline_coefficients.json').write_text(json.dumps(dict(fields=baseline,degrees=[1,1],condition_number=cond0),indent=2)+'\n')
heldout=[]
for label,ns,ys,ts in [('density',[.25,.26],[.20,.28],[.7,1.6]),('composition',[.19,.23],[.31,.32],[.7,1.6]),('temperature',[.19,.23],[.20,.28],[2.5,3.]),('joint',[.25,.26],[.31,.32],[2.5,3.])]:
 for n in ns:
  for Y in ys:
   for T in ts:heldout.append((label,n,T,Y))
baseline_errors=[]
for label,n,T,Y in heldout:
 budget();s=exact.state(n,T,Y);H,detail=implicit_hessian(s)
 gate(f'{label} exact forward residual {n,T,Y}',s.max_normalized_equation_residual,2e-8)
 prediction=evaluate(n,T,Y,coeff);small=evaluate(n,T,Y,baseline)
 # Independent FD of the full reference entropy at decreasing T steps.
 cvs=[]
 for h in (1e-3,5e-4):
  budget();plus=exact.state(n,T+h,Y);minus=exact.state(n,T-h,Y)
  cvs.append(T*(plus.entropy_density-minus.entropy_density)/(2*h))
 gate(f'{label} reference Cv FD ladder {n,T,Y}',abs(cvs[0]/cvs[1]-1),2e-6)
 errors=dict(free_MeV_per_baryon=abs(prediction['free_per_baryon']-s.free_energy_density/n),energy_MeV_per_baryon=abs(prediction['energy_density']/n-s.energy_density/n),
             pressure_relative=abs(prediction['pressure']/s.pressure-1),entropy_relative=abs(prediction['entropy_density']/s.entropy_density-1),
             mu_n_MeV=abs(prediction['mu_n']-s.mu_n_mev),mu_p_MeV=abs(prediction['mu_p']-s.mu_p_mev),cv_relative=abs(prediction['cv_fixed_nY']/cvs[1]-1),
             hessian_frobenius_relative=float(np.linalg.norm(np.array(prediction['chemical_hessian'])-H)/np.linalg.norm(H)))
 thresholds=dict(free_MeV_per_baryon=.005,energy_MeV_per_baryon=.005,pressure_relative=.01,entropy_relative=.01,mu_n_MeV=.05,mu_p_MeV=.05,cv_relative=.01,hessian_frobenius_relative=.02)
 for k,v in errors.items():gate(f'{label} fit {k} {n,T,Y}',v,thresholds[k])
 gate(f'{label} true H positive {n,T,Y}',float(np.linalg.eigvalsh(H)[0]<=0),.1)
 gate(f'{label} fitted H positive {n,T,Y}',float(prediction['min_chemical_eigenvalue']<=0),.1)
 gate(f'{label} fitted Cv positive {n,T,Y}',float(prediction['cv_fixed_nY']<=0),.1)
 # Identities are built into the single potential, not independent accuracy.
 gate(f'{label} fitted Euler identity {n,T,Y}',abs((1-Y)*n*prediction['mu_n']+Y*n*prediction['mu_p']-prediction['free_energy_density']-prediction['pressure']),1e-10)
 gate(f'{label} fitted e=f+Ts {n,T,Y}',abs(prediction['energy_density']-prediction['free_energy_density']-T*prediction['entropy_density']),1e-10)
 baseline_errors.append(dict(block=label,n=n,T=T,Y=Y,free_error=abs(small['free_per_baryon']-s.free_energy_density/n),pressure_relative=abs(small['pressure']/s.pressure-1),hessian_relative=float(np.linalg.norm(np.array(small['chemical_hessian'])-H)/np.linalg.norm(H))))
 rows.append(dict(block=label,n=n,T=T,Y=Y,exact=s.asdict(),exact_H=H.tolist(),exact_min_eigenvalue=float(np.linalg.eigvalsh(H)[0]),exact_Cv=cvs[1],prediction=prediction,errors=errors));checkpoint('blocked_holdouts')
# Baseline failures retained at the same specified tolerances.
base_pass=all(r['free_error']<.005 and r['pressure_relative']<.01 and r['hessian_relative']<.02 for r in baseline_errors)
(P/'baseline_results.json').write_text(json.dumps(dict(passed=base_pass,heldouts=baseline_errors,thresholds=dict(free=.005,pressure=.01,hessian=.02)),indent=2)+'\n')
report=dict(passed=all(c['passed'] for c in checks),checks=checks,training=training,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths},domain=DOMAIN,training_grid=dict(n=[.18,.195,.21,.225,.24],Y=[.18,.22,.26,.30],T=[0.,2.]),holdout_protocol='entire density,composition,temperature slabs beyond corresponding training hulls plus joint corners; no random neighboring split',condition_number=cond,external_matched_finiteT_GM1_gate='OPEN')
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds'],'baseline',base_pass)
for c in checks:
 if not c['passed']:print('FAIL',c)
