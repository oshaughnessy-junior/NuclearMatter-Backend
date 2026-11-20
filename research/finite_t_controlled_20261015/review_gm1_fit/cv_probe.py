from pathlib import Path
import json,sys,time,math
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT/'gm1_potential_fit'),str(ROOT/'rmf_reference')]
import potential as fit
import gm1_finite_t as exact
start=time.monotonic();checks=[];rows=[]
for row in json.loads((P/'results.json').read_text())['rows']:
 n,T,Y=row['n'],row['T'],row['Y'];values=[]
 for h in [1e-3,5e-4]:
  plus,minus=exact.state(n,T+h,Y),exact.state(n,T-h,Y)
  checks.append(dict(name=f'Cv perturbed forward {n,T,Y,h}',error=max(plus.max_normalized_equation_residual,minus.max_normalized_equation_residual),tolerance=2e-8,passed=max(plus.max_normalized_equation_residual,minus.max_normalized_equation_residual)<2e-8))
  values.append(T*(plus.entropy_density-minus.entropy_density)/(2*h))
 ladder=abs(values[0]/values[1]-1);error=abs(fit.evaluate(n,T,Y)['cv_fixed_nY']/values[1]-1)
 checks.extend([dict(name=f'exact Cv ladder{T}',error=ladder,tolerance=2e-6,passed=ladder<2e-6),dict(name=f'newheldout Cv{T}',error=error,tolerance=.01,passed=error<.01)])
 rows.append(dict(n=n,T=T,Y=Y,reference_Cv_ladder=values,fitted_Cv=fit.evaluate(n,T,Y)['cv_fixed_nY']))
r=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start);(P/'cv_results.json').write_text(json.dumps(r,indent=2)+'\n');print(len(checks),r['passed'],r['runtime_seconds'])
