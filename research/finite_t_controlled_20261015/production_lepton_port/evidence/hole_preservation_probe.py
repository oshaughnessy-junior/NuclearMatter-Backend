import importlib.util,sys,math,json,hashlib
from pathlib import Path
from scipy.special import kve
from nmbackend.reference import finite_t_leptons as new
P=Path(__file__).resolve().parents[1]
path=P/'pre_hole_bound_repair/staged/src/nmbackend/reference/finite_t_leptons.py'
name='nmbackend.reference._before_hole_bound';spec=importlib.util.spec_from_file_location(name,path);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
rows=[]
for T,mu,m in [(new.M_E/z,a*new.M_E/z,new.M_E) for z in (40.,100.,300.,600.)for a in (0.,.53,-.53)]+[(.73,-.29,new.M_E),(13.7,91.3,new.M_MU),(1e-30,.37e-30,0.),(.001,180.,new.M_MU)]:
 a=old.state_mu(T,mu,m);b=new.state_mu(T,mu,m)
 assert a.asdict()==b.asdict()
 rows.append(dict(check='state_all_fields_bitwise_unchanged',T=T,mu=mu,m=m,pass_=True))
 try: before=old.thermal_energy_density(a)
 except ArithmeticError as exc:
  assert m==new.M_E and abs(mu/T)==.53 and m/T==600.
  after=new.thermal_energy_density(b)
  number=math.exp(2*math.log(m)+math.log(T)+math.log(kve(2,600.))-600.-math.log(math.pi**2*new.HBARC**3))
  expected=2*number*((3*T+m*kve(1,600.)/kve(2,600.))*math.cosh(mu/T)-m*abs(math.sinh(mu/T)))
  assert abs(after/expected-1)<3e-10
  v=m/T;pf=new.HBARC*(3*math.pi**2*abs(b.net_density))**(1/3)/T;xc=pf*pf/(math.hypot(pf,v)+v)
  bound=new._hole_log2_upper_bound(T,v,xc)
  assert bound<=-1075.
  rows.append(dict(check='old_functional_failure_resolved',old_error=str(exc),T=T,mu=mu,m=m,actual=after,expected=expected,relative_error=abs(after/expected-1),physical_log2_upper_bound=bound,physical_log_upper_bound=bound*math.log(2),pass_=True))
 else:
  after=new.thermal_energy_density(b);assert after==before
  rows.append(dict(check='successful_thermal_bitwise_unchanged',T=T,mu=mu,m=m,pass_=True))
# Extra ordinary positive hole controls: fresh returned states from both signs,
# several finite masses, no grids or parameter-domain acceptance claim.
for T,mu,m in [(1.,.37,0.),(.01277497375,.53*.01277497375,new.M_E),(17.,90.,new.M_MU)]:
 a=old.state_mu(T,mu,m);b=new.state_mu(T,mu,m)
 assert old.thermal_energy_density(a)==new.thermal_energy_density(b)
 rows.append(dict(check='retained_positive_hole_control_bitwise',T=T,mu=mu,m=m,pass_=True))
result=dict(passed=len(rows),failed=0,source_sha256=hashlib.sha256((P/'staged/src/nmbackend/reference/finite_t_leptons.py').read_bytes()).hexdigest(),rows=rows)
(P/'evidence/hole_preservation_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
