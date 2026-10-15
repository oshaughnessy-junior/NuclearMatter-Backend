"""Additional actual zero-reference refinement and immutable evidence guards."""
from pathlib import Path
import sys,json,hashlib,importlib.util,subprocess,ast,time
P=Path(__file__).parent;sys.dont_write_bytecode=True;start=time.monotonic();checks=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(abs(e)<=t)))
sp=importlib.util.spec_from_file_location('zero_refinement_detailed_source',P/'reference.py');R=importlib.util.module_from_spec(sp);sys.modules[sp.name]=R;sp.loader.exec_module(R);v=json.loads((P/'validation_results.json').read_text());rows=[]
for x in v['zero_diagnostics']:
 E=x['E'];coarse=R.detailed(E,R.COARSE);fine=x['reference'];errors={k:coarse['total'][k]/fine['total'][k]-1 for k in['rate','transport_rate','outgoing_energy_moment']}
 for k,e in errors.items():ck('zero actual source refinement '+str((E,k)),e,1e-4)
 e=(coarse['total']['neutrino_energy_change']-fine['total']['neutrino_energy_change'])/(fine['total']['rate']*2);ck('zero actual signedRT source refinement '+str(E),e,1e-4);rows.append(dict(E=E,coarse=coarse,fine=fine))
paths=[P/n for n in ['PREREGISTRATION.json','training_references.json','heldout_references.json','model.json','FITTING_LOCK.json','validation_results.json','QUALIFICATION.json']];before={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in paths}
for name,args in [('compute.py',['training']),('compute.py',['heldout']),('validate.py',[])]:
 done=subprocess.run([sys.executable,'-B',str(P/name),*args],capture_output=True,text=True,timeout=10);ck('refuse frozen evidence overwrite '+name+str(args),float(done.returncode==0 or'Refusing to overwrite frozen'not in done.stderr),0)
ck('frozen evidence bytes unchanged',float(before!={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in paths}),0)
for f in P.glob('*.py'):ast.parse(f.read_text())
ck('all own Python syntax',0,0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],zero_reference_rows=rows,seconds=time.monotonic()-start);(P/'final_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','seconds']},indent=2))
