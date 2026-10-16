from pathlib import Path
import sys,importlib.util,json,hashlib,math
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_lepton_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_leptons as L
s=importlib.util.spec_from_file_location('nmbackend.reference.thermal_old_diagnostic',R/'review_production_lepton_compatibility/reviewed_compatibility_candidate/src/nmbackend/reference/finite_t_leptons.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());rows=[]
for i in range(4,8):
 ref=lock['states'][i];out=dict(index=i,oracle=ref,outcomes={})
 for name,mod in [('old',old),('new',L)]:
  st=mod.state_mu(ref['T'],ref['mu'],ref['mass']);log=[];original=mod._integrate
  def spy(fun,*args,**kw):
   try:r=original(fun,*args,**kw);log.append(dict(component=fun.__name__,success=True));return r
   except ArithmeticError as e:log.append(dict(component=fun.__name__,success=False,exception=str(e)));raise
  mod._integrate=spy
  try:v=mod.thermal_energy_density(st);out['outcomes'][name]=dict(value=v,exception=None)
  except ArithmeticError as e:out['outcomes'][name]=dict(value=None,exception=str(e))
  finally:mod._integrate=original
  out['outcomes'][name]['trace']=log
 rows.append(out)
(P/'diagnostic_results.json').write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),initial_failure_unchanged=hashlib.sha256((P/'results.json').read_bytes()).hexdigest()==json.loads((P/'INITIAL_FAILURE_PIN.json').read_text())['results.json']),indent=2)+'\n')
print(json.dumps([dict(index=r['index'],outcomes=r['outcomes'])for r in rows],indent=2))
