"""Saved-only positive band arithmetic; never substitutes for spectral calls."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy.special import logsumexp
P=Path(__file__).resolve().parent;L=json.loads((P/'LOCK.json').read_text());lockh=hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest()
HC=197.3269804;rows={};checks=[]
def positive(log):
 if log is None:return {'value':0.,'log':None,'reason':'absent_initial_neutrino_occupation'}
 v=math.exp(log);return {'value':v if v else None,'log':log,'reason':'represented' if v else 'underflow_positive'}
def add(name,error,gate,kind,acceptance=True):checks.append({'name':name,'error':float(error),'gate':gate,'passed':bool(abs(error)<gate),'kind':kind,'acceptance_gate':acceptance})
for job in L['jobs']:
 p=P/'raw'/f'{job["id"]:03d}.json';r=json.loads(p.read_text())
 if not r['complete'] or r['job']!=job or r['lock_sha256']!=lockh or r['candidate_sha256']!=L['candidate_sha256']:raise ArithmeticError('Incomplete/changed exact node identity')
 v=r['result']
 if v['E_MeV']!=job['E_MeV'] or v['rule']!=job['kernel_rule'] or v['bath']!=L['bath']:raise ArithmeticError('Changed physical state/rule')
 rows[job['id']]=r
 for key in ('max_shell_relative','max_lab_energy_scaled','max_unsquared_rest_delta_scaled','literal_current_reciprocity'):
  add(str(job['id'])+' '+key,v[key],1e-10,'kinematics')
 if job['type']=='control':
  base=next(x['result'] for x in rows.values() if x['job']['type']=='baseline' and x['job']['incident_order']==32 and x['job']['incident_index']==job['baseline_incident_index'])
  for component in ('log_j_s-1','log_true_chi_s-1'):
   add(str(job['id'])+' '+job['tag']+' '+component,math.expm1(v[component]-base[component]),3e-6,'edge_control')
results={}
for n in (12,20,32):
 values=[x for x in rows.values() if x['job']['type']=='baseline' and x['job']['incident_order']==n]
 if len(values)!=n:raise ArithmeticError('Missing positive band node')
 results[str(n)]={}
 for radiation in ('empty','FD_mu0'):
  results[str(n)][radiation]={}
  for direction in ('forward','backward'):
   result={}
   for power,name in ((0,'number'),(1,'energy')):
    terms=[]
    for r in values:
     E=r['job']['E_MeV'];v=r['result'];logD=2*math.log(E)-math.log(2*math.pi**2*HC**3)
     if radiation=='empty' and direction=='backward':continue
     x=E/2
     occ=(0. if radiation=='empty' else -float(np.logaddexp(0.,-x))) if direction=='forward' else -float(np.logaddexp(0.,x))
     component=v['log_j_s-1'] if direction=='forward' else v['log_true_chi_s-1']
     terms.append(math.log(r['job']['weight_MeV'])+logD+component+occ+power*math.log(E))
    result[name]=positive(float(logsumexp(terms)) if terms else None)
   results[str(n)][radiation][direction]=result
for lo,hi in ((12,20),(20,32)):
 for rad in ('empty','FD_mu0'):
  for direction in ('forward','backward'):
   for name in ('number','energy'):
    a=results[str(lo)][rad][direction][name];b=results[str(hi)][rad][direction][name]
    if a['log'] is None or b['log'] is None:
     if a['reason']!='absent_initial_neutrino_occupation' or b['reason']!=a['reason']:raise ArithmeticError('Missing positive branch disguised as absence')
     error=0.
    else:error=math.expm1(b['log']-a['log'])
    add(f'{rad} {direction} {name} order{lo}to{hi}',error,3e-6,'incident_refinement',acceptance=lo==20)
for name,h in L['copied_sources'].items():
 if hashlib.sha256((P/name).read_bytes()).hexdigest()!=h:raise ArithmeticError('Frozen source drift')
out={'status':'AUTHOR_SELECTED_BAND_PASS_PENDING_DIFFERENT_REVIEW' if all(c['passed'] for c in checks if c['acceptance_gate']) else 'FAIL','physical_calls':len(rows),'band_MeV':[1.1,1.4],'positive_branch_results':results,'checks':checks,'failed_checks_preserved':[x for x in checks if not x['passed']],'max_edge_control_relative':max(abs(c['error']) for c in checks if c['kind']=='edge_control'),'source_lock_sha256':lockh,'one_helicity_DOS':'E²/(2pi²HC³) included once','units':{'number':'fm^-3 s^-1','energy':'MeV fm^-3 s^-1'},'whole_energy_qualified':False,'other_bands_or_event_total_parity':False,'unrun_axes':['phi'],'rigorous_uniform_tail_bound':'OPEN','independent_integrated_phase_and_incident_review':'PENDING'}
(P/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],'checks',len(checks),'failed',len(out['failed_checks_preserved']),'maxedge',out['max_edge_control_relative'])
for c in checks:
 if c['kind']=='incident_refinement':print(c['name'],c['error'],'PASS' if c['passed'] else 'FAIL_RETAINED')
