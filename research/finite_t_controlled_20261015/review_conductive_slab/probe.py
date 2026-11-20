"""Independent RK2/integrated-conductivity-face nonlinear heat review."""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
p=slab.Physics();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(stage):
 (H/'results.json').write_text(json.dumps(dict(stage=stage,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')

# Independent method of lines matrix-exponential sine mode, different N/mode.
N=13;dx=1/N;x=(np.arange(N)+.5)/N;A=np.diag(np.full(N,-2.))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1);A[0,0]=A[-1,-1]=-3;A/=dx*dx
for mode in [1,3]:
 v=np.sin(mode*math.pi*x);lam=4*math.sin(mode*math.pi/(2*N))**2/dx**2
 out=expm(A*.017)@v
 gate('heldout sine matrix mode'+str(mode),float(np.max(abs(out-v*math.exp(-lam*.017)))),2e-13)
rows.append(dict(analytic_matrix_N=N,duration=.017))

# Actual independently integrated conductivity potential with new reservoirs.
left,right=1.06,.95
G=lambda t:quad(lambda z:p.k(z),right,t,epsabs=1e-12,epsrel=2e-10)[0]
flux=G(left);ss=[]
for xi in [.07,.31,.58,.91]:
 theta=brentq(lambda z:G(z)-flux*(1-xi),right,left,xtol=3e-13)
 h=1e-4;tp=brentq(lambda z:G(z)-flux*(1-xi-h),right,left,xtol=3e-13);tm=brentq(lambda z:G(z)-flux*(1-xi+h),right,left,xtol=3e-13)
 q=-p.k(theta)*(tp-tm)/(2*h)
 gate('heldout steady integral/gradient '+str(xi),q/flux-1,2e-7)
 ss.append(dict(x=xi,theta=theta,flux_by_gradient=q))
rows.append(dict(steady_flux=flux,steady_samples=ss));save('analytic and steady')

def nonlinear(steps):
 start=time.monotonic();N=5;dx=1/N;dt=.006/steps
 x=(np.arange(N)+.5)/N;theta=1+.025*np.sin(2*math.pi*x)+.015*np.cos(math.pi*x)
 initial=theta.copy();W=np.array([p.W(z)for z in theta]);E0=float(W.mean());S0=float(np.mean([p.entropy(z)for z in theta]))
 accum=np.zeros(4);gx,gw=leggauss(3)
 # Independent solve of actual u, rather than author's inverse method.
 def invert(W):return np.array([brentq(lambda z:p.W(z)-v,.9,1.1,xtol=2e-13)for v in W])
 def stage(W):
  if time.monotonic()-start>110:raise TimeoutError('Independent nonlinear bounded experiment')
  t=invert(W);q=[];allT=np.r_[left,t,right];positions=np.r_[0.,x,1.]
  for a,b,d in zip(allT[:-1],allT[1:],np.diff(positions)):
   # Integral of actual κ across each face; differs from author harmonic κ.
   v=np.sum(gw*np.array([p.k(float((a+b)/2+(b-a)*z/2))for z in gx]))*(b-a)/2
   q.append(-v/d)
  q=np.array(q);rhs=(q[:-1]-q[1:])/dx
  bS=q[0]/left-q[-1]/right
  prod=float(np.sum(q*np.diff(1/allT)))
  return rhs,np.array([q[0],q[-1],bS,prod]),t,q
 history=[]
 for i in range(steps):
  k1,a1,_,_=stage(W);trial=W+dt*k1;k2,a2,_,_=stage(trial)
  W=W+dt*(k1+k2)/2;accum+=dt*(a1+a2)/2
  t=invert(W);E=float(np.mean([p.W(z)for z in t]));S=float(np.mean([p.entropy(z)for z in t]))
  history.append(dict(time=(i+1)*dt,energy_residual=E-E0-accum[0]+accum[1],total_entropy=S-S0-accum[2],entropy_face_discrepancy=S-S0-accum[2]-accum[3]))
 return dict(theta=t.tolist(),initial_theta=initial.tolist(),steps=steps,N=N,duration=.006,history=history,initial_energy=E0,initial_entropy=S0,final_energy=E,final_entropy=S,boundary_in=float(accum[0]),boundary_out=float(accum[1]),boundary_entropy=float(accum[2]),integrated_face_production=float(accum[3]),runtime=time.monotonic()-start)

for n in [12,24]:
 r=nonlinear(n);rows.append(dict(nonlinear=r));save('nonlinear'+str(n))
 gate('independent actual EOS energy ledger'+str(n),max(abs(h['energy_residual'])for h in r['history']),3e-10)
 gate('independent entropy total sign'+str(n),min(0,min(h['total_entropy']for h in r['history'])),1e-12)
 gate('independent bounded state'+str(n),int(min(r['theta'])<right or max(r['theta'])>left),0)
 gate('positive incoming reservoir energy'+str(n),min(0,r['boundary_in']),0)
 gate('positive outgoing reservoir energy'+str(n),min(0,r['boundary_out']),0)
r1=rows[-2]['nonlinear'];r2=rows[-1]['nonlinear'];e1=abs(r1['history'][-1]['entropy_face_discrepancy']);e2=abs(r2['history'][-1]['entropy_face_discrepancy'])
gate('RK2 entropy independent temporal residual decreases',max(0,e2/e1-.5),0)
rows.append(dict(entropy_residual_ratio=e2/e1,nonlinear_max_theta_step_difference=max(abs(np.array(r1['theta'])-r2['theta']))))
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
