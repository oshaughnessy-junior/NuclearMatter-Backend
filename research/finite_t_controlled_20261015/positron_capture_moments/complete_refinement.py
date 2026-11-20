from pathlib import Path
import json,time,math
import moments as M
import conditional as C
P=Path(__file__).resolve().parent;old=json.loads((P/'initial_refinement_partial.json').read_text());rows=old['rows'];checks=old['checks'];start=time.monotonic();b,meta=M.PC.matched(2,1e-7,.3)
def save():
 (P/'refinement.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),seconds=time.monotonic()-start),indent=2)+'\n')
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for control in [dict(neutron_order=28,positron_order=56,initial_order=8,outgoing_order=8,phi_order=8,tail=60.)]:
 r=M.integrate(b,**control);rows.append(dict(kind='total',result=r));save();gate('exact neutron FD forward density',r['neutron_density']/7e-8-1,3e-8)
 for name,s in r['scenarios'].items():
  gate('energy ledger '+name,s['energy_ledger_scaled'],3e-12);gate('direct energy ledger '+name,s['direct_ledger_scaled'],3e-12)
  if name!='empty':
   gate('KMS '+name,s['max_event_kms_error'],3e-11);gate('entropy nonnegative '+name,int(s['entropy']< -1e-22),0)
   if name=='LTE':gate('LTE net',s['net']['number']/s['forward']['number']['value'],3e-11)
 save()
# New comparisons keep the same3e-6 gate. Coarse study.py failures remain separate.
for i,j in [(0,1),(1,2),(1,3)]:
 a=rows[i]['result'];z=rows[j]['result']
 for name in z['scenarios']:
  for direction in ['forward','backward']:
   for key in M.NAMES:
    v=z['scenarios'][name][direction][key]['value'];u=a['scenarios'][name][direction][key]['value']
    if v:gate('qualified refine '+str((i,name,direction,key)),u/v-1,3e-6)
 save()
for controls in [dict(norder=20,corder=12,eorder=16,phi_order=8,tail=40.),dict(norder=28,corder=16,eorder=24,phi_order=12,tail=60.)]:
 r=C.spectrum(b,5.,**controls);rows.append(dict(kind='conditional',result=r));save()
cs=[r['result']for r in rows if r['kind']=='conditional'];ref=json.loads((M.R/'positron_capture/refinement.json').read_text())
# accepted final source values frozen in note/raw; no normalization fit.
for key,target in [('j',.3302220190349698),('chi',.987407803020526)]:
 gate('conditional refinement '+key,cs[0][key]['value']/cs[1][key]['value']-1,3e-6);gate('conditional absolute accepted response '+key,cs[1][key]['value']/target-1,3e-6)
save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
