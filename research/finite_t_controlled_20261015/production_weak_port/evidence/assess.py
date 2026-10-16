from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parents[1];R=P.parent;checks=[]
def check(name,error,tol):
 e=float(error);checks.append(dict(name=name,error=e,tolerance=tol,passed=bool(math.isfinite(e)and abs(e)<=tol)))
def ratio(name,value,reference,tol=3e-6):
 if reference==0:check(name,value,0.)
 else:check(name,value/reference-1,tol)
fixture=json.loads((P/'staged/tests/data/free_weak_reference.json').read_text())
maxima={'saved_RQ':0.,'base_fine_all_moments':0.,'fresh_reference_all_moments':0.,'isolated_tail_all_moments':0.,'energy_ledger_gross':0.,'direct_energy_ledger_gross':0.,'FD_entropy_affinity_gross':0.}
for case in range(3):
 for channel in['EC','PC','NEUTRON_DECAY']:
  data=json.loads((P/'evidence'/f'candidate_{case}_{channel}.json').read_text());assert data['complete']is True
  for name,row in data['profiles']['fine'].items():
   coarse=data['profiles']['base'][name]
   for branch in['forward','backward']:
    for i in range(6):
     a,b=coarse[branch][i]['value'],row[branch][i]['value'];err=0. if a==b==0 else a/b-1
     check(f'{case}/{channel}/{name}/{branch}/{i}/basefine',err,3e-6);maxima['base_fine_all_moments']=max(maxima['base_fine_all_moments'],abs(err))
     if case==0 and i<2:
      target=fixture['channels'][channel]['saved'][name]['F'if branch=='forward'else'B'][i];err=0. if a==b==target==0 else b/target-1
      check(f'saved/{channel}/{name}/{branch}/{i}',err,3e-6);maxima['saved_RQ']=max(maxima['saved_RQ'],abs(err))
   for profile in['base','fine']:
    rr=data['profiles'][profile][name];grossR=rr['forward'][0]['value']+rr['backward'][0]['value'];grossQ=rr['forward'][1]['value']+rr['backward'][1]['value']
    for field,mx in [('energy_ledger_scaled','energy_ledger_gross'),('direct_energy_ledger_scaled','direct_energy_ledger_gross')]:
     check(f'{case}/{channel}/{profile}/{name}/{field}',rr[field],3e-9);maxima[mx]=max(maxima[mx],abs(rr[field]))
    d=dict(rr['diagnostics']);check(f'{case}/{channel}/{profile}/{name}/four_shell',d['max_four_residual']/939.56542052,2e-10)
    for field in['max_mass_relative']:
     if field in d:check(f'{case}/{channel}/{profile}/{name}/{field}',d[field],2e-10)
    counts=dict(rr['particle_stoichiometry']);check('baryon count '+str((case,channel,profile,name)),counts['neutron']+counts['proton'],0.);check('charge count '+str((case,channel,profile,name)),counts['proton']-counts['electron_net'],0.);check('Le count '+str((case,channel,profile,name)),counts['electron_net']+counts.get('nue',0)-counts.get('antinue',0),0.)
    if rr['radiation']['mode']=='FD':
     check(f'{case}/{channel}/{profile}/{name}/directKMS',rr['max_event_kms_error'],2e-9)
     b=rr['bath'];eq=b['mu_p']+b['mu_e']-b['mu_n'];eq=eq if channel=='EC'else-eq;mur=rr['radiation']['mu_nue'if channel=='EC'else'mu_antinue'];expected=rr['net'][0]['value']*(eq-mur)/b['T'];err=(rr['entropy_production']['value']-expected)/grossR
     check(f'{case}/{channel}/{profile}/{name}/entropy_affinity',err,3e-9);maxima['FD_entropy_affinity_gross']=max(maxima['FD_entropy_affinity_gross'],abs(err));check(f'{case}/{channel}/{profile}/{name}/Hpositive',min(0.,rr['entropy_production']['value'])/grossR,0.)
     if name=='LTE':
      check(f'{case}/{channel}/{profile}/LTE_R',rr['net'][0]['value']/grossR,2e-10);check(f'{case}/{channel}/{profile}/LTE_Q',rr['net'][1]['value']/grossQ,2e-10)
  if case:
   reference=json.loads((P/'evidence'/f'reference_{case}_{channel}.json').read_text());assert reference['complete']is True
   for name,ref in reference['profiles']['fine'].items():
    cand=data['profiles']['fine'][name]
    for bi,branch in enumerate(['forward','backward']):
     for i in range(6):
      if channel=='PC':rv=ref['raw']['sums'][ref['selected_scenario_index']][bi][i]
      else:
       oldname=['number','neutrino_energy','electron_energy','proton_energy','neutron_energy','material_energy_loss'][i];rv=ref['scenarios'][name][branch][oldname]['value']
      cv=cand[branch][i]['value'];err=0. if cv==rv==0 else cv/rv-1;check(f'fresh/{case}/{channel}/{name}/{branch}/{i}',err,3e-6);maxima['fresh_reference_all_moments']=max(maxima['fresh_reference_all_moments'],abs(err))
   # Actual incoming particle density is evaluated from phase-space distributions.
   cand=data['profiles']['fine']['empty'];ref=reference['profiles']['fine']['empty']
   cd=dict(cand['diagnostics'])['parent_particle_density'];rd=ref['raw']['neutron_density']if channel=='PC'else ref['proton_density']if channel=='EC'else ref['parent_density'];ratio(f'fresh/{case}/{channel}/particle_density',cd,rd,3e-6)
for ch in['EC','PC','NEUTRON_DECAY']:
 fine=json.loads((P/'evidence'/f'candidate_2_{ch}.json').read_text())['profiles']['fine'];tail=json.loads((P/'evidence'/f'tail_{ch}.json').read_text())['rows']
 for name,row in tail.items():
  for branch in['forward','backward']:
   for i in range(6):
    a,b=fine[name][branch][i]['value'],row[branch][i]['value'];err=0. if a==b==0 else a/b-1;check(f'isolatedtail/{ch}/{name}/{branch}/{i}',err,3e-6);maxima['isolated_tail_all_moments']=max(maxima['isolated_tail_all_moments'],abs(err))
for source,expected in json.loads((P/'evidence/SOURCE_PIN.json').read_text()).items():check('unchanged source '+source,int(hashlib.sha256((R/source).read_bytes()).hexdigest()!=expected),0.)
guards=[]
for old in(P/'pre_signed_value_guard').glob('candidate_*.json'):
 new=P/'evidence'/old.name
 if new.exists():
  a=json.loads(old.read_text())['profiles'];b=json.loads(new.read_text())['profiles'];guards.append(dict(file=old.name,old_new_profiles_bitwise_json_equal=a==b));check('draft_signed_value_guard_same_actual '+old.name,int(a!=b),0.)
anchors=json.loads((P/'evidence/anchors.json').read_text());assert anchors['failed']==0
output=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),maxima=maxima,guard_draft_identity=guards,physical_scope='same free full-current model; method/source agreement is not empirical or model uncertainty',anchor_pass=anchors['passed'])
(P/'evidence/ASSESSMENT.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items()if k not in['checks','guard_draft_identity']},indent=2))
