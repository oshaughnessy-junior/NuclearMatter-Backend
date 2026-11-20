from pathlib import Path
import sys,json,math
P=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference.finite_t_rmf import state_density
T,n,Y=14.,.19,.22;s=state_density(T,n,Y);rows=[]
for e in [.001,.0005]:
 a=state_density(T*(1+e),n,Y);b=state_density(T*(1-e),n,Y);cv=(a.energy_density-b.energy_density)/(2*e*T);phi=(a.phi-b.phi)/(2*e*T);entropy=(a.free_energy_density-b.free_energy_density)/(2*e*T);rows.append(dict(step=e,plus=a.asdict(),minus=b.asdict(),Cv_FD=cv,phi_T_FD=phi,minus_entropy_FD=entropy))
last=rows[-1];prev=rows[-2];checks=[]
for field,want in [('Cv_FD',s.cv_fixed_density_composition),('phi_T_FD',s.phi_temperature_derivative),('minus_entropy_FD',-s.entropy_density)]:
 got=(4*last[field]-prev[field])/3;err=abs(got/want-1);checks.append(dict(name=field,extrapolated=got,analytic=want,relative_error=err,gate=3e-6,passed=err<3e-6))
(P/'evidence/response_probe.json').write_text(json.dumps(dict(state=s.asdict(),rows=rows,checks=checks),indent=2)+'\n');print(checks)
