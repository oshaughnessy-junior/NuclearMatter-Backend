"""Isolated missing-export admission and stable algebra checks, not SciPy1.7."""
import ast,importlib.util,sys,json,math,hashlib
from pathlib import Path
import numpy as np
import scipy.special as special
import nmbackend.reference
P=Path(__file__).resolve().parents[1]
source=P/'staged/src/nmbackend/reference/finite_t_leptons.py'
old=P/'pre_scipy17_repair/staged/src/nmbackend/reference/finite_t_leptons.py'
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
oldmod=load(old,'nmbackend.reference._pre_compat')
original=special.log_expit
rows=[]
try:
 del special.log_expit
 try: load(old,'nmbackend.reference._old_missing_export')
 except ImportError as exc:
  rows.append(dict(check='historical_missing_export_failure_reproduced',pass_=True,error=str(exc)))
 else: raise AssertionError('Old candidate unexpectedly imported')
 newmod=load(source,'nmbackend.reference._compatible_missing_export')
 rows.append(dict(check='repaired_import_without_scipy_log_expit',pass_=True))
finally:
 special.log_expit=original
for x in [0.,-0.,1e-300,-1e-300,1e-12,-1e-12,1.,-1.,40.,-40.,600.,-600.,1000.,-1000.,1e300,-1e300]:
 actual=float(newmod._log_expit(x))
 expected=-math.log1p(math.exp(-x)) if x>=0 else x-math.log1p(math.exp(x))
 assert actual<=0 and math.isfinite(actual)
 assert actual==expected or abs(actual/expected-1)<2e-15
 rows.append(dict(check='signed_extreme_helper',input=x,actual=actual,expected=expected,pass_=True))
for T,mu,m in [(newmod.M_E/z,a*newmod.M_E/z,newmod.M_E) for z in (40.,100.,300.,600.) for a in (0.,.53,-.53)]+[(.73,-.29,newmod.M_E),(13.7,91.3,newmod.M_MU),(1e-30,.37e-30,0.),(.001,180.,newmod.M_MU)]:
 a=oldmod.state_mu(T,mu,m);b=newmod.state_mu(T,mu,m)
 for key,value in a.asdict().items():
  actual=getattr(b,key);relative=abs(actual/value-1) if value else abs(actual-value)
  assert relative<3e-11,(T,mu,key,relative)
  rows.append(dict(check='unchanged_numerical_state',T=T,mu=mu,m=m,field=key,relative=relative,pass_=True))
 outcomes=[]
 for mod,state in [(oldmod,a),(newmod,b)]:
  try: outcomes.append(dict(value=mod.thermal_energy_density(state)))
  except ArithmeticError as exc: outcomes.append(dict(error=str(exc)))
 if 'error' in outcomes[0] or 'error' in outcomes[1]:
  assert outcomes[0]==outcomes[1],(T,mu,outcomes)
  rows.append(dict(check='unchanged_thermal_rejection_not_forward_acceptance',T=T,mu=mu,m=m,
                   error=outcomes[0]['error'],pass_=True))
 else:
  relative=abs(outcomes[1]['value']/outcomes[0]['value']-1)
  assert relative<3e-11
  rows.append(dict(check='unchanged_thermal_increment',T=T,mu=mu,m=m,relative=relative,pass_=True))
imports=[node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.ImportFrom) and node.module=='scipy.special']
assert [a.name for n in imports for a in n.names]==['expit']
rows.append(dict(check='special_imports_only_expit',pass_=True))
result=dict(passed=len(rows),failed=0,simulation_is_not_scipy17_execution=True,
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows)
(P/'evidence/compat_import_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
