import json,math,importlib.util,sys,hashlib
from pathlib import Path
from nmbackend.reference import finite_t_leptons as new
P=Path(__file__).resolve().parents[1];snap=P/'pre_unit_hole_repair'
lock=json.loads((snap/'review/VALIDATION_LOCK.json').read_text())
name='nmbackend.reference._e0c16_before_unit_hole';path=snap/'author/staged/src/nmbackend/reference/finite_t_leptons.py'
spec=importlib.util.spec_from_file_location(name,path);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
rows=[]
for item in lock['states']+lock['additional_valid_states']:
 T,mu,m=item['T'],item['mu'],item['mass'];a=old.state_mu(T,mu,m);b=new.state_mu(T,mu,m);assert a.asdict()==b.asdict()
 rows.append(dict(check='all_state_fields_bitwise',T=T,mu=mu,m=m,pass_=True))
 try:before=old.thermal_energy_density(a)
 except ArithmeticError as exc:before=None;old_error=str(exc)
 captured=[];original=new._integrate
 def observe(fun,breaks,rtol,epsabs):
  val=original(fun,breaks,rtol,epsabs)
  if fun.__name__=='hole_integrand':captured.append(float(val))
  return val
 new._integrate=observe
 try:after=new.thermal_energy_density(b)
 finally:new._integrate=original
 if 'thermal'in item:
  rel=abs(after/item['thermal']-1);assert rel<3e-10
  rows.append(dict(check='locked_total_thermal',T=T,mu=mu,m=m,actual=after,reference=item['thermal'],relative_error=rel,prior_exception=old_error if before is None else None,pass_=True))
 if before is not None:
  rel=abs(after/before-1);assert rel<3e-10
  rows.append(dict(check='prior_success_preservation',T=T,mu=mu,m=m,bitwise=after==before,relative_change=rel,pass_=True))
 if 'hole_float'in item:
  if captured:
   v=m/T;pf=new.HBARC*(3*math.pi**2*abs(b.net_density))**(1/3)/T;xc=pf*pf/(math.hypot(pf,v)+v)
   logfactor=4*math.log(T)+math.log(new._MEASURE)+float(new._log_expit(xc-(abs(mu)-m)/T))+2.5*math.log(xc)+.5*math.log(xc+2*v)+math.log(xc+v)
   hole=new._restore(captured[0],logfactor)
  else:hole=0.
  assert hole==item['hole_float'],(T,hole,item['hole_float'])
  rows.append(dict(check='locked_component_restore',T=T,mu=mu,m=m,kernel=captured[0]if captured else None,actual_hole=hole,expected_hole=item['hole_float'],pass_=True))
r=dict(passed=len(rows),failed=0,source_sha256=hashlib.sha256((P/'staged/src/nmbackend/reference/finite_t_leptons.py').read_bytes()).hexdigest(),rows=rows)
(P/'evidence/unit_hole_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='rows'}))
