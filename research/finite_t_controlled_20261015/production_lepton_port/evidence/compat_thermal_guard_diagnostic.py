"""Read-only numeric diagnostic; no candidate fix or acceptance widening."""
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import kve
from nmbackend.constants import M_E,HBARC
from nmbackend.reference.finite_t_leptons import state_mu,thermal_energy_density,_log_expit
P=Path(__file__).resolve().parents[1]
rows=[]
for alpha in [.53,-.53]:
 T=M_E/600.;mu=alpha*T;s=state_mu(T,mu,M_E)
 v=M_E/T;edge=(abs(mu)-M_E)/T
 pf=HBARC*(3*math.pi**2*abs(s.net_density))**(1/3)/T
 cold_edge=pf*pf/(math.hypot(pf,v)+v)
 log_h=float(_log_expit(cold_edge-edge))
 def hole(x):return math.sqrt(x*(x+2*v))*(v+x)*(cold_edge-x)*math.exp(float(_log_expit(x-edge))-log_h)
 value,error,info=quad_vec(hole,0.,cold_edge,epsabs=0.,epsrel=2e-12,norm='max',limit=500,full_output=True)
 try: thermal=thermal_energy_density(s);outcome={'value':thermal}
 except ArithmeticError as exc:outcome={'error':str(exc)}
 # Independent leading small-surface estimate of hole magnitude, only a
 # diagnostic scale: 4/15*v*sqrt(2*v)*xc**(5/2). Evaluate in logs.
 log_hole=(math.log(4/15)+math.log(v)+.5*math.log(2*v)+2.5*math.log(cold_edge))
 rows.append(dict(T=T,mu=mu,m=M_E,alpha=alpha,state=s.asdict(),cold_edge=cold_edge,
                  hole_midpoint_value=hole(cold_edge/2),hole_quadrature=float(value),
                  hole_error=float(error),quad_status=int(info.status),quad_success=bool(info.success),
                  quad_message=info.message,estimated_log_hole=log_hole,
                  float64_min_subnormal_log=math.log(float(np.nextafter(0.,1.))),
                  thermal_outcome=outcome,
                  total_energy_density_before_cold_subtraction=s.energy_density,
                  cold_reference_rest_energy=M_E*abs(s.net_density),
                  reconstructed_thermal_from_resolved_bulk=s.energy_density-M_E*abs(s.net_density),
                  classical_bessel_thermal=(2*math.exp(2*math.log(M_E)+math.log(T)+math.log(kve(2,600.))-600.-math.log(math.pi**2*HBARC**3))
                    *((3*T+M_E*kve(1,600.)/kve(2,600.))*math.cosh(alpha)-M_E*abs(math.sinh(alpha))))))
(P/'evidence/compat_thermal_guard_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='state'}for row in rows],indent=2))
