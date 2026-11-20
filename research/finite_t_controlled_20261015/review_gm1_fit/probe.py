from pathlib import Path
import sys,json,math,time,hashlib,atexit
import numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'gm1_potential_fit'),str(ROOT/'rmf_reference'),str(ROOT/'rmf_phase')]
import potential as fit
import gm1_finite_t as exact
from stability import implicit_hessian
start=time.monotonic();checks=[];rows=[];phase='initial';c=json.loads((ROOT/'gm1_potential_fit/coefficients.json').read_text())['fields']
paths=[ROOT/'gm1_potential_fit'/x for x in ['potential.py','coefficients.json','fit_study.py','results.json','baseline_results.json']]+[ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'rmf_phase/stability.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(x['passed']for x in checks)and end==hashes,phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
# Value-only manual Chebyshev recurrence, independent of author's chebder/chainrule.
def recurrence(x,n):
 values=[1.,x]
 for k in range(2,n):values.append(2*x*values[-1]-values[-2])
 return values[:n]
def value(n,T,Y):
 x=(n-.22)/.04;d=(1-2*Y)**2;z=(2*d-(.34**2+.64**2))/(.64**2-.34**2)
 xx=recurrence(x,5);zz=recurrence(z,4)
 fields={k:sum(c[k][i][j]*xx[i]*zz[j]for i in range(5)for j in range(4))for k in c}
 return 939.+fields['cold']+T*T*fields['T2']+T**4*fields['T4']
def free(nn,np_,T):return (nn+np_)*value(nn+np_,T,np_/(nn+np_))
def chemical(pair,T):
 out=[]
 for k in range(2):
  z=np.array(pair,dtype=complex);z[k]+=1e-20j;out.append(float(np.imag(free(*z,T))/1e-20))
 return np.array(out)
def independent_H(pair,T,rel):
 H=np.zeros((2,2))
 for k in range(2):
  h=rel*pair[k];plus=pair.copy();minus=pair.copy();plus[k]+=h;minus[k]-=h;H[:,k]=(chemical(plus,T)-chemical(minus,T))/(2*h)
 return H
points=[(.2537,2.73,.3237),(.1811,.12,.1817),(.2593,2.99,.3291)]
old=json.loads((ROOT/'gm1_potential_fit/results.json').read_text());existing={(x['n'],x['T'],x['Y'])for x in old['rows']}|{(x['n'],T,x['Y'])for x in old['training']for T in [0.,2.]}
for n,T,Y in points:
 if time.monotonic()-start>105:raise TimeoutError('Review105s bound')
 phase=f'point{n,T,Y}';gate('newcoordinates '+str((n,T,Y)),float((n,T,Y)in existing),.5)
 s=exact.state(n,T,Y);gate('exact forward constraints '+str((n,T,Y)),s.max_normalized_equation_residual,2e-8)
 H,detail=implicit_hessian(s);p=fit.evaluate(n,T,Y);pair=np.array([n*(1-Y),n*Y]);mu=chemical(pair,T)
 gate('manual complex potential value '+str(T),abs(value(n,T,Y)-p['free_per_baryon']),2e-10)
 gate('manual complex chemical '+str(T),np.max(abs(mu-[p['mu_n'],p['mu_p']])),2e-9)
 pressure=n*n*np.imag(value(n+1e-20j,T,Y))/1e-20
 entropy=-np.imag(n*value(n,T+1e-20j,Y))/1e-20
 gate('manual complex pressure '+str(T),abs(pressure/p['pressure']-1),2e-10)
 gate('manual complex entropy '+str(T),abs(entropy/p['entropy_density']-1),2e-10)
 for h in [2e-4,1e-4]:
  HH=independent_H(pair,T,h);gate('manual complex/densityFD Hessian '+str((T,h)),np.linalg.norm(HH-np.array(p['chemical_hessian']))/np.linalg.norm(HH),2e-7)
 # Exact re-solved densityFD Hessian; no author derivative code reused.
 HH=np.zeros((2,2))
 for k in range(2):
  h=pair[k]*1e-4;pp=pair.copy();mm=pair.copy();pp[k]+=h;mm[k]-=h
  sp=exact.state(float(pp.sum()),T,float(pp[1]/pp.sum()));sm=exact.state(float(mm.sum()),T,float(mm[1]/mm.sum()))
  gate('perturbed primal plus '+str((T,k)),sp.max_normalized_equation_residual,2e-8);gate('perturbed primal minus '+str((T,k)),sm.max_normalized_equation_residual,2e-8)
  HH[:,k]=(np.array([sp.mu_n_mev,sp.mu_p_mev])-np.array([sm.mu_n_mev,sm.mu_p_mev]))/(2*h)
 gate('exact FD vs exact implicit H '+str(T),np.linalg.norm(HH-H)/np.linalg.norm(H),3e-6)
 errors=dict(free=abs(p['free_per_baryon']-s.free_energy_density/n),energy=abs(p['energy_density']/n-s.energy_density/n),pressure=abs(p['pressure']/s.pressure-1),entropy=abs(p['entropy_density']/s.entropy_density-1),mu_n=abs(p['mu_n']-s.mu_n_mev),mu_p=abs(p['mu_p']-s.mu_p_mev),hessian=float(np.linalg.norm(np.array(p['chemical_hessian'])-HH)/np.linalg.norm(HH)),affinity_MeV=(p['mu_n']-p['mu_p'])-(s.mu_n_mev-s.mu_p_mev))
 errors['balance_factor']=abs(math.expm1(errors['affinity_MeV']/T))
 for name,tol in [('free',.005),('energy',.005),('pressure',.01),('entropy',.01),('mu_n',.05),('mu_p',.05),('hessian',.02),('balance_factor',.01)]:gate('newheldout accuracy '+str(T)+' '+name,errors[name],tol)
 gate('newheldout affinity '+str(T),abs(errors['affinity_MeV']),.01)
 gate('exact local scalar branch '+str(T),float(detail['canonical_scalar_curvature_fm_minus2']<=0),.5)
 gate('exact/fitted full H local positive '+str(T),float(min(np.linalg.eigvalsh(H)[0],p['min_chemical_eigenvalue'])<=0),.5)
 rows.append(dict(n=n,T=T,Y=Y,exact=s.asdict(),prediction=p,independent_complex_chemical=mu.tolist(),independent_exact_FD_H=HH.tolist(),exact_implicit_H=H.tolist(),errors=errors,scalar_curvature=detail['canonical_scalar_curvature_fm_minus2']));save()
phase='boundary_controls'
for n,T,Y in [(.18,0.,.18),(.26,0.,.33),(.18,3.,.33),(.26,3.,.18)]:
 p=fit.evaluate(n,T,Y)
 gate('domain endpoint finite '+str((n,T,Y)),float(not all(math.isfinite(x)for x in [p['pressure'],p['entropy_density'],p['cv_fixed_nY'],p['min_chemical_eigenvalue']])),.5)
 if T==0:gate('cold entropy/Cv vanish '+str((n,Y)),max(abs(p['entropy_density']),abs(p['cv_fixed_nY'])),1e-15)
for point in [(.1799,1.,.2),(.2601,1.,.2),(.2,-1e-8,.2),(.2,3.0001,.2),(.2,1.,.1799),(.2,1.,.3301),(.2,math.nan,.2)]:
 try:fit.evaluate(*point)
 except ValueError:gate('domain rejects '+str(point),0.,.5)
 else:gate('domain rejects '+str(point),1.,.5)
phase='complete';save();print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),runtime=time.monotonic()-start,failed=[c for c in checks if not c['passed']]),indent=2))
