from pathlib import Path
import json,math,hashlib,datetime
P=Path(__file__).resolve().parent;checks=[]
def add(name,error,gate):checks.append({'name':name,'error':float(error),'gate':gate,'pass':bool(error<gate)})
tags=['OLD_TAIL70','OLD_TAIL90','E120_PANEL','E120_SEMI16','E120_SEMI24','E130_PANEL','E130_SEMI16','E130_SEMI24']
raw={tag:json.loads((P/(tag+'.json')).read_text()) for tag in tags}
for tag,r in raw.items():
 if not r['complete']:raise ArithmeticError('Unresolved physical call: '+tag)
 if r['candidate_sha256']!=hashlib.sha256((P/'conditional_beta.py').read_bytes()).hexdigest():raise ArithmeticError('Raw candidate source identity drift')
a=raw['OLD_TAIL70']['result'];b=raw['OLD_TAIL90']['result']
old_change=math.expm1(b['j']['log']-a['j']['log'])
if not abs(old_change)>3e-6:raise ArithmeticError('Expected original failed witness not reproduced')
for E,prefix in ((1.2,'E120'),(1.3,'E130')):
 ref=raw[prefix+'_SEMI24']['result']
 for suffix in ('PANEL','SEMI16','SEMI24'):
  r=raw[prefix+'_'+suffix]['result']
  if r['E_MeV']!=E or r['bath']!=ref['bath']:raise ArithmeticError('State identity drift')
  for component in ('log_j_s-1','log_true_chi_s-1'):
   add(f'{prefix}_{suffix}_{component}',abs(math.expm1(r[component]-ref[component])),3e-6)
  for key in ('max_shell_relative','max_lab_energy_scaled','max_unsquared_rest_delta_scaled','literal_current_reciprocity'):
   add(f'{prefix}_{suffix}_{key}',r[key],1e-10)
  add(f'{prefix}_{suffix}_actual_products_KMS',abs(r['KMS_actual_products']),2e-9)
for r in json.loads((P/'PHASE_JACOBIAN_RAW.json').read_text()):
 if r['h']==.0003:add(str(r['E'])+'_absolute_phase_Jacobian',abs(r['relative_error']),3e-6)
for original,h in json.loads((P/'ORIGINAL_SOURCE_PIN.json').read_text()).items():
 if hashlib.sha256(Path(original).read_bytes()).hexdigest()!=h:raise ArithmeticError('Published original modified')
 if hashlib.sha256((P/'beta_pr55_reference'/Path(original).name).read_bytes()).hexdigest()!=h:raise ArithmeticError('Current source-copy drift')
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AUTHOR_SELECTED_PASS_PENDING_DIFFERENT_REVIEW' if all(x['pass'] for x in checks) else 'FAIL','checks':checks,'fresh_profile_calls':8,'old_failed_tail_witness':{'relative_j_change':old_change,'gate':3e-6,'pass':False,'preserved':True},'separate_angular_axes':'OPEN; profiles couple inner orders','upper_incident_panel_order_convergence':'OPEN; not rerun','whole_energy_qualification':False,'independent_integrated_phase_review':'PENDING'}
(P/'ASSESSMENT.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks; preserved original failure',old_change)
if any(not x['pass'] for x in checks):raise SystemExit(1)
