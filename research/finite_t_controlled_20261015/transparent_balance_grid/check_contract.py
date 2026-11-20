from pathlib import Path
import importlib.util,sys,math,json,hashlib
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('three_bath_contract_only',P/'grid.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for args in [(2.,1e-7,.3),(1.,1e-6,.3),(True,1e-7,.3),(1.,float('nan'),.3),(1.,1e-7,True),(1.,1e-7,.01),(1.,1e-7,float('inf'))]:
 try:m.material(*args);ok=False
 except ValueError:ok=True
 ck('reject outside selected experiment '+str(args),ok)
for refined,tail in [(1,False),(False,1),(None,False),(False,float('nan'))]:
 try:m.evaluate(1.,1e-7,.3,refined,tail);ok=False
 except ValueError:ok=True
 ck('reject nonliteral quadrature flag '+str((refined,tail)),ok)
for T,n in m.STATES:
 b,s=m.material(T,n,.5);ck('fresh selected EOS baryon root '+str((T,n)),abs(s['neutron']['density']/(n*.5)-1)<1e-10 and abs(s['proton']['density']/(n*.5)-1)<1e-10);ck('fresh selected EOS charge '+str((T,n)),abs(s['electron']['net_density']/(n*.5)-1)<1e-10)
for name,h in json.loads((P/'SOURCE_PIN.json').read_text()).items():ck('source/oldanchor unchanged '+name,hashlib.sha256((P.parent/name).read_bytes()).hexdigest()==h)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']]);(P/'contract_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
