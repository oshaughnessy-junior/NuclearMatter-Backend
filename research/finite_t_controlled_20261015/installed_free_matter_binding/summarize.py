from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent;R=P.parent
base=json.loads((P/'weak_base.json').read_text());fine=json.loads((P/'weak_fine.json').read_text());cycle=json.loads((P/'cycle_summary.json').read_text());old=json.loads((R/'three_source_cooling_element/results.json').read_text())['rows'][0]['records'][0]['source'];checks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
for channel in ['EC','PC','beta']:
 for field in ['number','neutrino_energy']:
  a=base['source']['channels'][channel][field];b=fine['source']['channels'][channel][field];ck('actual source refinement '+channel+field,a/b-1,3e-6);ck('accepted prior samebath absolute '+channel+field,a/old['channels'][channel][field]-1,3e-6)
for row in [base,fine]:
 s=row['state'];z=row['source'];Q=z['Q'];ndot=row['ndot'];scale=Q
 ck('instant first law/Q '+row['profile'],(row['energy_dot']-row['firstlaw_energy_dot'])/scale,2e-9)
 # Isolate weak heating after subtracting actual mechanical term and composition;
 # this explicitly avoids normalizing weak source cancellation by rest energy.
 weakdot=s['cv']*row['T_dot']-((s['energy']+s['pressure'])/s['nB']-s['e_n'])*ndot+s['e_Y']*row['Ye_dot'];ck('thermal weak plus composition/Q '+row['profile'],(weakdot+Q)/Q,2e-9)
# Pure binding admission checks preserve forward numerical qualification separately.
import sys;sys.path.insert(0,str(P));import matter as M
for args in [(True,1e-7,.3),(2.,0.,.3),(2.,1e-7,0.),(2.,1e-7,.05),(2.,1e-7,.9),(float('nan'),1e-7,.3)]:
 try:M.state(*args)
 except ValueError:ck('invalid binding state '+str(args),0.,0.)
 else:ck('invalid binding state '+str(args),1.,0.)
for val in [None,0,1,'LTE']:
 try:M.state(2.,1e-7,.3,photons=val)
 except ValueError:ck('literal closure '+str(val),0.,0.)
 else:ck('literal closure '+str(val),1.,0.)
counts={}
for fn in ['forward_results.json','partial_results.json','cycle_32.json','cycle_64.json','cycle_summary.json','weak_base.json','weak_fine.json']:
 z=json.loads((P/fn).read_text());counts[fn]=dict(passed=z['passed'],failed=len(z['failed']))
summary=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,accepted_probes=counts,total_current_passed=sum(v['passed']for v in counts.values())+sum(c['passed']for c in checks),initial_failures={'wrong_sign_check_identity':8,'JSON_numpy_count_reporter':1,'LOCK_literal_timestamp_typo':1},max_individual_source_refinement=max(c['error']for c in checks if c['name'].startswith('actual source refinement')),weak_instantaneous={k:fine[k]for k in ['ndot','Ye_dot','T_dot','EMpair_sources','photon_comparison']},cycle={k:cycle[k]for k in ['T_min','T_max','gross_absolute_work64','work64','return_delta_u64','quarter_work64']})
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status'],summary['passed'],'total',summary['total_current_passed'],summary['failed'])
