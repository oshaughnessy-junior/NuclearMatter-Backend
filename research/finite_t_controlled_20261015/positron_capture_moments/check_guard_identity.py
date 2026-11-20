"""Post-study empty-scenario guard leaves valid physical calculations unchanged."""
from pathlib import Path
import json
import moments as M
P=Path(__file__).resolve().parent;b,_=M.PC.matched(2,1e-7,.3);old=json.loads((P/'results.json').read_text())['rows'][0];new=M.integrate(b,neutron_order=8,positron_order=16,initial_order=8,outgoing_order=8,phi_order=8,tail=40.);checks=[]
for name in old['scenarios']:
 for direct in ['forward','backward']:
  for quantity in M.NAMES:
   a=old['scenarios'][name][direct][quantity];z=new['scenarios'][name][direct][quantity];checks.append(dict(name='valid branch unchanged '+str((name,direct,quantity)),passed=bool(a==z)))
 checks.append(dict(name='valid net unchanged '+name,passed=bool(old['scenarios'][name]['net']==new['scenarios'][name]['net'])))
checks.append(dict(name='forward density unchanged',passed=bool(old['neutron_density']==new['neutron_density'])))
(P/'guard_identity.json').write_text(json.dumps(dict(checks=checks,passed=all(x['passed']for x in checks),seconds=new['seconds'],meaning='only rejects an empty scenario list; unchanged valid numerical core, not an accuracy replay'),indent=2)+'\n');print(len(checks),all(x['passed']for x in checks))
