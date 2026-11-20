"""Independent finite displacements of the single fitted potential and grid scan."""
from potential import *
import json,time
start=time.monotonic();checks=[]
fit=json.loads((P/'coefficients.json').read_text());assert 'delta²' in fit['form']
c=fit['fields']
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for n,T,Y in [(.22,1.2,.25),(.252,2.7,.322)]:
 a=evaluate(n,T,Y,c);dens=np.array([n*(1-Y),n*Y]);H=np.array(a['chemical_hessian'])
 for rel in (2e-4,1e-4):
  numerical_H=np.zeros((2,2))
  for j in range(2):
   h=rel*dens[j];v=np.zeros(2);v[j]=h
   def point(d):return evaluate(float(sum(d)),T,float(d[1]/sum(d)),c)
   plus,minus=point(dens+v),point(dens-v)
   mu=(plus['free_energy_density']-minus['free_energy_density'])/(2*h)
   gate(f'fd canonical mu{j} {n,T,Y}h{rel}',abs(mu-a[['mu_n','mu_p'][j]]),2e-5)
   numerical_H[:,j]=np.array([plus['mu_n']-minus['mu_n'],plus['mu_p']-minus['mu_p']])/(2*h)
  gate(f'fd canonical Hessian {n,T,Y}h{rel}',float(np.linalg.norm(numerical_H-H)/np.linalg.norm(H)),2e-7)
 for h in (1e-3,5e-4):
  plus,minus=evaluate(n,T+h,Y,c),evaluate(n,T-h,Y,c)
  entropy=-(plus['free_energy_density']-minus['free_energy_density'])/(2*h)
  Cv=(plus['energy_density']-minus['energy_density'])/(2*h)
  gate(f'fd potential entropy {n,T,Y}h{h}',abs(entropy/a['entropy_density']-1),2e-6)
  gate(f'fd potential Cv {n,T,Y}h{h}',abs(Cv/a['cv_fixed_nY']-1),2e-6)
 for key in ('cold','T2','T4'):gate(f'analytic isospin evenness {key}n{n}',abs(field(c[key],n,Y)-field(c[key],n,1-Y)),1e-12)
minimum_H=math.inf;minimum_Cv_over_nT=math.inf
for n in np.linspace(*DOMAIN['n'],9):
 for Y in np.linspace(*DOMAIN['Y'],9):
  for T in np.linspace(.1,3.,7):
   a=evaluate(float(n),float(T),float(Y),c);minimum_H=min(minimum_H,a['min_chemical_eigenvalue']);minimum_Cv_over_nT=min(minimum_Cv_over_nT,a['cv_fixed_nY']/(n*T))
gate('567node fit-only chemical stability scan',float(minimum_H<=0),.1)
gate('567node fit-only thermal stability scan',float(minimum_Cv_over_nT<=0),.1)
for args in [(.179,1.,.25),(.22,-.1,.25),(.22,3.1,.25),(.22,1.,.331),(math.nan,1.,.25)]:
 try:evaluate(*args,c)
 except ValueError:gate('explicit domain rejection '+str(args),0.,.1)
 else:gate('explicit domain rejection '+str(args),1.,.1)
result=dict(passed=all(x['passed'] for x in checks),checks=checks,fit_only_grid_nodes=567,min_fitted_chemical_eigenvalue=minimum_H,min_fitted_Cv_over_nT=minimum_Cv_over_nT,runtime_seconds=time.monotonic()-start,scope='Analytic implementation/fit-only stability checks; identities and positive interpolation scan do not establish exactGM1 accuracy throughoutdomain')
(P/'analytic_results.json').write_text(json.dumps(result,indent=2)+'\n');print(sum(x['passed'] for x in checks),'/',len(checks),result['runtime_seconds'])
for x in checks:
 if not x['passed']:print('FAIL',x)
