"""Blocked cold and low-T checks with frozen fitted coefficients."""
from potential import *
import sys,json,math,time
sys.path.insert(0,str(P.parent/'rmf_reference'));import gm1_finite_t as exact
start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for n,Y in [(.248,.315),(.258,.325)]:
 cold=exact.cold_state(n,Y);v=evaluate(n,0.,Y)
 for name,error,tol in [('cold freeabs',abs(v['free_per_baryon']-cold.free_energy_density/n),.005),('cold pressure',abs(v['pressure']/cold.pressure-1),.01),('cold mu_n',abs(v['mu_n']-cold.mu_n_mev),.05),('cold mu_p',abs(v['mu_p']-cold.mu_p_mev),.05),('cold entropy',abs(v['entropy_density']),1e-15),('cold Cv',abs(v['cv_fixed_nY']),1e-15)]:gate(str((n,Y))+name,error,tol)
 mass=cold.dirac_mass_mev/exact.HBARC
 gamma=sum(exact.kf(z*n)*math.hypot(exact.kf(z*n),mass) for z in (1-Y,Y))/(3*exact.HBARC)
 T=.05;warm=exact.state(n,T,Y);pred=evaluate(n,T,Y)
 gate(str((n,Y))+'lowT exact residual',warm.max_normalized_equation_residual,2e-8)
 gate(str((n,Y))+'lowT fit entropy',abs(pred['entropy_density']/warm.entropy_density-1),.01)
 gate(str((n,Y))+'T2 coefficient versus cold DOS gamma',abs((-2*n*field(json.loads((P/'coefficients.json').read_text())['fields']['T2'],n,Y))/gamma-1),.01)
 rows.append(dict(n=n,Y=Y,cold=cold.asdict(),cold_prediction=v,lowT=warm.asdict(),lowT_prediction=pred,gamma_cold=gamma))
report=dict(passed=all(c['passed'] for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start)
(P/'cold_limit_results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds'])
