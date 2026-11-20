"""Isolated total-source tail control after selected fine event refinement."""
from pathlib import Path
import json
import electron_capture as C
HERE=Path(__file__).resolve().parent;bath,state=C.matched();initial=json.loads((HERE/'results.json').read_text())['rules'][-1]
new=C.integrate(bath,proton_order=28,electron_order=48,initial_order=12,outgoing_order=12,phi_order=12,tail=80.)
(HERE/'tail_raw.json').write_text(json.dumps(new,indent=2)+'\n');checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol)))
ck('tail80 actual proton density',new['proton_density']/3e-8-1,3e-8)
for scenario in new['scenarios']:
 for direction in ['forward','backward']:
  for name in C.NAMES:
   a=initial['scenarios'][scenario][direction][name]['value'];b=new['scenarios'][scenario][direction][name]['value']
   if b:ck(str((scenario,direction,name))+' isolatedtail60/80',a/b-1,3e-6)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],seconds=new['seconds'])
(HERE/'tail_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
