"""Preserve initial coarse failures and add stricter positive pilot controls."""
from pathlib import Path
import json, math
import pc_kernel as pc
H=Path(__file__).parent;original=json.loads((H/'initial_coarse_results.json').read_text());rows=[];checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(phase):
 (H/'refinement.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks),initial_failed_controls='initial_coarse_results.json'),indent=2)+'\n')
b=pc.crossed.Bath(**original['rows'][0]['bath']);event=[r['result']for r in original['rows']if r.get('kind')=='CM_events'][-1]
for order in [16,20]:
 r=pc.integrate(b,5.,lepton_order=order,angle_order=12,energy_order=20,phi_order=4,tail=50.,outer_tail=24.)
 rows.append(dict(label='response'+str(order),result=r));save('order'+str(order))
for key in ['j','chi']:
 gate('outer16vs20 '+key,rows[0]['result'][key]['value']/rows[1]['result'][key]['value']-1,3e-6)
 gate('independent CM vs response20 '+key,rows[1]['result'][key]['value']/event[key]['value']-1,3e-6)
# Refine outer cutoff, inner quadrature and angular resolution together, then
# retain the exact controls so no KMS identity is mistaken for convergence.
r=pc.integrate(b,5.,lepton_order=20,angle_order=16,energy_order=28,phi_order=8,tail=70.,outer_tail=32.)
rows.append(dict(label='inner_angle_tail_refinement',result=r));save('refined')
for key in ['j','chi']:
 gate('inner_angle_tail '+key,r[key]['value']/rows[1]['result'][key]['value']-1,3e-6)
 gate('final independent absolute '+key,r[key]['value']/event[key]['value']-1,3e-6)
gate('direct KMS',r['log_kms_actual']-r['log_kms_expected'],3e-11)
save('complete');print(json.dumps(dict(checks=len(checks),failures=[c for c in checks if not c['passed']])))
