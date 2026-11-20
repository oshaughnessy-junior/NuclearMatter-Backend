"""Corrected join contract: real frozen rows, synthetic mutations only."""
from pathlib import Path
import importlib.util,json,copy,math,hashlib
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
spec=importlib.util.spec_from_file_location('repaired_join',R/'antinu_source_join/join.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
h=json.loads((R/'hot_beta_moments/results.json').read_text());p=json.loads((R/'positron_capture_moments/refinement.json').read_text());state=h['rows'][0]['matched_state'];ps=json.loads((R/'positron_capture_moments/MATCHED_STATE.json').read_text());hs=next(x['result']['scenarios']for x in h['rows']if x.get('result',{}).get('controls',{}).get('rest_order')==32);pr=next(x['result']['scenarios']for x in p['rows']if x.get('kind')=='total'and x['result']['controls']['initial_order']==12)
checks=[];rows=[]
def reject(name,fn,reason=None):
 try:fn();rejected=False;message='accepted'
 except(ValueError,ArithmeticError)as e:message=str(e);rejected=True
 checks.append(dict(name=name,passed=rejected and(reason is None or reason in message),rejected=rejected,reason=message))
def gate(name,error,tolerance):checks.append(dict(name=name,error=float(error),tolerance=tolerance,passed=bool(math.isfinite(error)and abs(error)<=tolerance)))
gate('exact material dictionaries match',int(state!=ps),0)
for key in ['empty','LTE']:
 a,b=hs[key],pr[key];out=J.combine(a,b,state,ps);rows.append(dict(key=key,arithmetic_only_not_independent_physics_review=out))
 for direction in ['forward','backward','net']:
  for name in ['number','neutrino_energy','material_energy_loss']:
   vals=[x[direction][name]if direction=='net'else x[direction][name]['value']for x in [a,b]]
   gate('plain arithmetic '+key+' '+direction+' '+name,out[direction][name]-sum(vals),2e-20)
 gate('Le conservation '+key,out['sources']['matter_Le_dt']+out['sources']['radiation_Le_dt'],0)
 gate('charge conservation '+key,out['sources']['dn_p_dt']-out['sources']['dn_e_net_dt'],0)
 scale=out['forward']['neutrino_energy']+out['backward']['neutrino_energy']
 gate('direct material/radiation energy '+key,out['direct_energy_residual']/scale,3e-10)
 if key=='LTE':gate('LTE gross scaled roundoff',out['gross_normalized_net'],3e-12)
a=copy.deepcopy(hs['LTE']);b=copy.deepcopy(pr['LTE']);a['scenario']['temperature']=2.;b['scenario']['temperature']=3.
reject('explicit different radiation T',lambda:J.combine(a,b,state,ps),'Unsupported radiation fields')
a=copy.deepcopy(hs['empty']);b=copy.deepcopy(pr['empty']);b['net']['number']=float('nan')
reject('net NaN',lambda:J.combine(a,b,state,ps),'Nonfinite net')
b=copy.deepcopy(pr['empty']);b['net']['neutrino_energy']=float('inf')
reject('net infinity',lambda:J.combine(hs['empty'],b,state,ps),'Nonfinite net')
b=copy.deepcopy(pr['empty']);b['forward']['number'].update(value=None,representable=False)
reject('gross positive unrepresented',lambda:J.combine(hs['empty'],b,state,ps),'must be represented')
reject('actual -.8T vs -.4T occupations',lambda:J.combine(hs['emission_dominated'],pr['noneq'],state,ps),'Radiation occupations differ')
bad=copy.deepcopy(ps);bad['electron']['chemical_potential']+=1e-12
reject('changed physical electron root',lambda:J.combine(hs['empty'],pr['empty'],state,bad),'Material states')
reject('accepted-status empty required provenance',lambda:J.verify_manifest(dict(status='accepted_selected_sources',sha256={})), 'coverage missing')
sha={n:'0'*64 for n in J.REQUIRED};first='hot_beta_moments/results.json';sha={first:'f'*64,**{n:v for n,v in sha.items()if n!=first}}
reject('mutated source hash metadata',lambda:J.verify_manifest(dict(status='accepted_selected_sources',sha256=sha)),'Changed source/evidence '+first)
sha={'../outside-campaign': '0'*64,**{n:'0'*64 for n in J.REQUIRED}}
reject('unsafe provenance path',lambda:J.verify_manifest(dict(status='accepted_selected_sources',sha256=sha)),'Invalid source path')
sha={first:'nothex',**{n:'0'*64 for n in J.REQUIRED if n!=first}}
reject('malformed source digest',lambda:J.verify_manifest(dict(status='accepted_selected_sources',sha256=sha)),'Invalid source path or SHA256')
(H/'postrepair_results.json').write_text(json.dumps(dict(tested_source_sha256=hashlib.sha256((R/'antinu_source_join/join.py').read_bytes()).hexdigest(),checks=checks,rows=rows,all_pass=all(c['passed']for c in checks),qualification='integration contract only; independent channel reviews required'),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
