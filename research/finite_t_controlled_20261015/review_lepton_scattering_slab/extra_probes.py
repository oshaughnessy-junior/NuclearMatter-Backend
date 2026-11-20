"""Additional self-mode, hole and nonuniform streaming checks, no author writes."""
from pathlib import Path
import sys,json,types,math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import expit
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'lepton_scattering_slab';sys.dont_write_bytecode=True
B=types.ModuleType('extra_frozen_slab');B.__file__=str(AUTHOR/'slab.py');exec(compile((HERE/'reviewed_initial_slab.py').read_bytes(),str(HERE/'reviewed_initial_slab.py'),'exec'),B.__dict__)
g=json.loads((HERE/'reviewed_initial_accepted_results.json').read_text())['grids'][-1];E=np.array(g['E']);W=np.array(g['W']);mu=np.array(g['mu']);A=np.array(g['A']);checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
f=expit((.81-E)/(2.7+.7*mu));q=B.collision(g,f)
clone=dict(g);a=A.copy();np.fill_diagonal(a,np.max(A)*17.);clone['A']=a.tolist();qq=B.collision(clone,f)
ck('self-mode activities cannot alter paired populations',np.max(abs(q['df']-qq['df'])),0)
ck('self-mode activities cannot alter bath heat',q['heat']-qq['heat'],0)
ck('self-mode activities cannot alter entropy',q['entropy']-qq['entropy'],0)
events=A*f[:,None]*(1-f[None,:]);net=events.sum(0)-events.sum(1)
unblocked=A*f[:,None];wrong=unblocked.sum(0)-unblocked.sum(1)
ck('actual Pauli holes nontrivial at test populations',float(np.max(abs(net-wrong))/np.max(abs(net))<1e-3),0)
for chemical in [-2.17,.97]:
 occ=expit((chemical-E)/2);df,led=B.rhs(g,np.tile(occ,(4,1)),occ,occ)
 ck('new spatial+collision LTE '+str(chemical),np.max(abs(df)),3e-11)
 ck('new equal-boundary Nflux '+str(chemical),(led[0]-led[1])/led[0],3e-14)
 ck('new equal-boundary Eflux '+str(chemical),(led[2]-led[3])/led[2],3e-14)
# Arbitrary nonuniform populations: exact matrix exponential of each upwind chain.
nc=4;cross=.037;duration=.009;left=expit((-.24-E)/2.9);right=expit((-.81-E)/1.4)
initial=np.array([expit((-.31+.17*z-E)/(1.5+.2*z+.2*mu))for z in range(nc)])
def fun(t,y):return B.rhs(g,y.reshape(initial.shape),left,right,crossing_time=cross,collisions=False)[0].ravel()
sol=solve_ivp(fun,[0,duration],initial.ravel(),method='RK45',rtol=2e-12,atol=1e-17,max_step=.0002)
assert sol.success
exact=np.zeros_like(initial)
for a in range(len(E)):
 speed=abs(mu[a])*nc/cross;Q=-speed*np.eye(nc)
 for z in range(nc-1):Q[z+1,z]=speed
 data=initial[:,a]if mu[a]>0 else initial[::-1,a]
 boundary=left[a]if mu[a]>0 else right[a]
 prediction=boundary+expm(Q*duration)@(data-boundary)
 exact[:,a]=prediction if mu[a]>0 else prediction[::-1]
ck('fresh nonuniform streaming exact matrix-exponential solution',np.max(abs(exact-sol.y[:,-1].reshape(initial.shape))),3e-11)
result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],pauli_hole_relative_effect=float(np.max(abs(net-wrong))/np.max(abs(net))),streaming_exact=exact.tolist(),streaming_numerical=sol.y[:,-1].reshape(initial.shape).tolist())
(HERE/'extra_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','failed','pauli_hole_relative_effect']}))
