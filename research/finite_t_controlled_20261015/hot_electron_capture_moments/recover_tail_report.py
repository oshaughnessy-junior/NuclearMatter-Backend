"""Recover gates from already completed, preserved tail80 physical source."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;a=json.loads((HERE/'results.json').read_text())['rules'][-1];b=json.loads((HERE/'tail_raw.json').read_text());checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol)))
ck('tail80 actual proton density',b['proton_density']/3e-8-1,3e-8)
for scenario in b['scenarios']:
 for direction in ['forward','backward']:
  for name in ['number','neutrino_energy','electron_energy','proton_energy','neutron_energy','material_energy_loss']:
   x=a['scenarios'][scenario][direction][name]['value'];y=b['scenarios'][scenario][direction][name]['value']
   if y:ck(str((scenario,direction,name))+' isolatedtail60/80',x/y-1,3e-6)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],seconds=b['seconds'],report_recovery_only=True,raw_source='tail_raw.json')
(HERE/'tail_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=b['seconds'])))
