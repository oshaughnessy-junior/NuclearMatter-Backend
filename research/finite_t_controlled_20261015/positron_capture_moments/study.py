from pathlib import Path
import sys,json,time,math,hashlib,atexit
import moments as M
P=Path(__file__).resolve().parent;checks=[];rows=[];phase='start';start=time.monotonic()
def save():
 (P/'results.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,passed=all(c['passed']for c in checks),seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
b,meta=M.PC.matched(2,1e-7,.3);(P/'MATCHED_STATE.json').write_text(json.dumps(meta,indent=2)+'\n')
for order in [8,12,16]:
 phase='order '+str(order);r=M.integrate(b,neutron_order=order,positron_order=2*order,initial_order=8,outgoing_order=8,phi_order=8,tail=40.);rows.append(r);save()
 gate('forward neutron density '+str(order),r['neutron_density']/7e-8-1,3e-6);gate('event energy '+str(order),r['max_single_event_energy_residual'],3e-10);gate('event shell '+str(order),r['max_mass_residual'],3e-12)
 for name,s in r['scenarios'].items():
  gate('energy ledger '+str((order,name)),s['energy_ledger_scaled'],3e-12);gate('direct energy ledger '+str((order,name)),s['direct_ledger_scaled'],3e-12)
  if name!='empty':
   gate('event KMS '+str((order,name)),s['max_event_kms_error'],3e-11);gate('entropy sign '+str((order,name)),int(s['entropy']< -1e-22),0)
   if name=='LTE':gate('LTE relative net '+str(order),s['net']['number']/s['forward']['number']['value'],3e-11)
for a,b in zip(rows[:-1],rows[1:]):
 for name in b['scenarios']:
  for direction in ['forward','backward']:
   for key in M.NAMES:
    x=a['scenarios'][name][direction][key]['value'];y=b['scenarios'][name][direction][key]['value']
    if y:gate('refine '+str((a['controls']['neutron_order'],b['controls']['neutron_order'],name,direction,key)),x/y-1,3e-6)
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
