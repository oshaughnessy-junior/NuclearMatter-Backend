"""Weak-off FD derivative, entropy and independent mechanical-work checks."""
from density_element import *
from scipy.integrate import solve_ivp
import json,time
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for T,n,Y,photons in [(2.,1e-4,.3,False),(2.1,1.02e-4,.29,False),(2.,1e-4,.3,True)]:
 s=state(T,n,Y,photons=photons)
 for h in (2e-4,1e-4):
  plus=state(T,n*(1+h),Y,photons=photons);minus=state(T,n*(1-h),Y,photons=photons)
  dn=(plus['u_MeV_per_baryon']-minus['u_MeV_per_baryon'])/(2*n*h)
  ds=(plus['entropy_per_baryon']-minus['entropy_per_baryon'])/(2*n*h)
  gate(f'genuineFD un T{T}Y{Y}ph{photons}h{h}',abs(dn/s['u_n_MeV_fm3']-1),3e-7)
  gate(f'Maxwell entropy_n T{T}Y{Y}ph{photons}h{h}',abs(ds/(-s['dP_dT_fixed_nY']/n**2)-1),3e-7)
 gate(f'independent EOS un versus pressure identity ph{photons}T{T}',abs((s['P_MeV_fm3']-T*s['dP_dT_fixed_nY'])/(n*n*s['u_n_MeV_fm3'])-1),3e-10)
# Integrate temperature from the true u_n/u_T work equation; do NOT impose
# constant entropy in this ODE. Then compare with a separate entropy-root path.
Y=.3;n0=1e-4;T0=2.;amplitude=.02
initial=state(T0,n0,Y);S=initial['entropy_per_baryon']
def rhs(theta,T):
 n=density(theta);s=state(float(T[0]),n,Y);xdot=amplitude*math.pi*math.sin(2*math.pi*theta)
 return [(s['P_MeV_fm3']/n-n*s['u_n_MeV_fm3'])/s['u_T']*xdot]
solution=solve_ivp(rhs,(0.,1.),[T0],method='DOP853',rtol=2e-10,atol=1e-11,dense_output=True)
if not solution.success:raise ArithmeticError(solution.message)
samples=[]
for theta in np.linspace(0,1,17):
 n=density(float(theta));T=float(solution.sol(theta)[0]);s=state(T,n,Y);other=temperature_at_entropy(S,n,Y,T0)
 gate(f'ODE independent entropy root theta{theta}',abs(T/other-1),2e-9)
 gate(f'ODE entropy conservation theta{theta}',abs(s['entropy_per_baryon']/S-1),2e-9)
 samples.append(dict(theta=float(theta),nB=n,T=T,u=s['u_MeV_per_baryon'],entropy=s['entropy_per_baryon'],P=s['P_MeV_fm3']))
# Positive compression and negative expansion work independently integrated.
up=adiabatic_move(T0,n0,Y,density(.5));down=adiabatic_move(up['T'],density(.5),Y,n0)
for label,x in [('compression',up),('expansion',down)]:gate('independent Pdn/n2 '+label,abs(x['work_energy']-x['work_pressure_integral'])/max(abs(x['work_pressure_integral']),1e-300),2e-9)
gate('closed history temperature returns',abs(down['T']/T0-1),2e-10)
gate('closed positive negative work ledger',abs(up['work_pressure_integral']+down['work_pressure_integral'])/abs(up['work_pressure_integral']),2e-9)
report=dict(passed=all(c['passed'] for c in checks),checks=checks,photons=False,weak=False,prescribed_history='nB=n0 exp(.02 sin²(pi theta)),theta=t/(2e-6 s)',samples=samples,compression=up,expansion=down,runtime_seconds=time.monotonic()-start,scope='One reversible weak-off prescribed-density cycle; independent EOS derivative/entropy/work checks, no hydro')
(P/'weak_off_results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
