"""Independent cell-average initialization, overlap projection and RK2 gate."""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
p=slab.Physics();checks=[];rows=[]
main=json.loads((R/'conductive_spatial_refinement/results.json').read_text());tm=json.loads((R/'conductive_spatial_refinement/time_separation_results.json').read_text())
saved={(r['run']['controls']['N'],r['run']['controls']['steps']):r['run']for d in [main,tm]for r in d['rows']if'run'in r}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(stage):
 (H/'results.json').write_text(json.dumps(dict(stage=stage,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
def project(values,N):
 values=np.asarray(values);M=len(values);out=[]
 for i in range(N):
  a,b=i/N,(i+1)/N
  out.append(sum(v*max(0,min(b,(j+1)/M)-max(a,j/M))for j,v in enumerate(values))*N)
 return np.array(out)
def norms(v):
 v=np.asarray(v);return dict(L1=float(np.linalg.norm(v,1)/len(v)),L2=float(np.linalg.norm(v)/math.sqrt(len(v))),Linf=float(np.linalg.norm(v,np.inf)))
# Independent adaptive cell integrals, rather than author's nested Gauss rules.
init={}
for N in [4,8]:
 init[N]=np.array([quad(lambda x:p.W(1+.04*math.cos(math.pi*x)),i/N,(i+1)/N,epsabs=1e-13,epsrel=2e-10)[0]*N for i in range(N)])
 gate('independent adaptive energy average '+str(N),float(np.max(abs(init[N]-saved[(N,16)]['target_initial_W']))),3e-11)
 gate('independent projected average '+str(N),float(np.max(abs(project(saved[(16,16)]['target_initial_W'],N)-init[N]))),3e-11)
rows.append(dict(independent_initial_energy={str(N):v.tolist()for N,v in init.items()}));save('initialization')

def evolve(N,steps):
 start=time.monotonic();dx=1/N;dt=.0015/steps;W=init[N].copy();E0=float(np.mean(W))
 def invert(values):return np.array([brentq(lambda t:p.W(t)-v,.95,1.05,xtol=2e-13)for v in values])
 t=invert(W);S0=float(np.mean([p.entropy(z)for z in t]));theta0=t.copy();integrated_production=0;hist=[]
 def rhs(values):
  if time.monotonic()-start>110:raise TimeoutError('Independent spatial review bounded')
  t=invert(values);k=np.array([p.k(z)for z in t]);q=np.zeros(N+1)
  q[1:-1]=-2*k[:-1]*k[1:]/(k[:-1]+k[1:])*np.diff(t)/dx
  production=float(np.sum(q[1:-1]*np.diff(1/t)))
  return (q[:-1]-q[1:])/dx,production
 for i in range(steps):
  f1,s1=rhs(W);f2,s2=rhs(W+dt*f1);W+=dt*(f1+f2)/2;integrated_production+=dt*(s1+s2)/2
  t=invert(W);energy=float(np.mean([p.W(z)for z in t]));entropy=float(np.mean([p.entropy(z)for z in t]))
  hist.append(dict(time=(i+1)*dt,energy_residual=energy-E0,entropy_increase=entropy-S0,face_entropy_integral=integrated_production))
 return dict(N=N,steps=steps,theta=t.tolist(),initial_theta=theta0.tolist(),W=[p.W(z)for z in t],history=hist,runtime=time.monotonic()-start)

for N in [4,8]:
 r=evolve(N,16);rows.append(dict(RK2=r));save('RK2 '+str(N))
 gate('independent RK2 energy '+str(N),max(abs(h['energy_residual'])for h in r['history']),3e-10)
 gate('independent RK2 entropy '+str(N),min(0,min(h['entropy_increase']for h in r['history'])),1e-12)
 gate('independent RK2 max principle '+str(N),int(min(r['theta'])<min(r['initial_theta'])or max(r['theta'])>max(r['initial_theta'])),0)
 # Independent time integrator checks Euler's Richardson extrapolated value.
 extrap=2*np.array(saved[(N,32)]['final_W'])-np.array(saved[(N,16)]['final_W'])
 delta=norms(np.array(r['W'])-extrap)
 for name,value in delta.items():gate('RK2 vs Euler Richardson '+str(N)+' '+name,value,3e-8)
 rows.append(dict(RK2_vs_Euler_Richardson=dict(N=N,error=delta)))
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
