"""Synthetic-only join adversaries; NOT accepted physics/source evidence."""
from pathlib import Path
import importlib.util,sys,json,copy,math,hashlib
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
spec=importlib.util.spec_from_file_location('join_under_review',R/'antinu_source_join/join.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
state=json.loads((R/'hot_beta_moments/results.json').read_text())['rows'][0]['matched_state'];eq=state['neutron']['chemical_potential']-state['proton']['chemical_potential']-state['electron']['chemical_potential'];checks=[];witnesses=[]
def rejection(name,fn):
 try:out=fn();reject=False;detail=repr(out)
 except (ValueError,ArithmeticError)as e:reject=True;detail=str(e)
 checks.append(dict(name=name,expected='reject',rejected=reject,passed=reject));witnesses.append(dict(name=name,detail=detail))
def scenario(mode='empty',mu=eq):return dict(mode=mode,mu_nu=mu)if mode=='FD'else dict(mode=mode)
def row(mode='empty'):
 return dict(scenario=scenario(mode),forward={k:dict(value=v,log=math.log(v),representable=True)for k,v in [('number',1.),('neutrino_energy',.5),('material_energy_loss',.5)]},backward={k:dict(value=0. if mode=='empty'else v,log=None if mode=='empty'else math.log(v),representable=True)for k,v in [('number',1.),('neutrino_energy',.5),('material_energy_loss',.5)]},net={k:0. if mode=='FD'else v for k,v in [('number',1.),('neutrino_energy',.5),('material_energy_loss',.5)]})
a=row('FD');b=copy.deepcopy(a);a['scenario']['temperature']=2.;b['scenario']['temperature']=3.
rejection('different radiation temperature at same chemical potential',lambda:J.combine(a,b,state,state))
a=row();b=copy.deepcopy(a);b['net']['number']=float('nan')
rejection('nonfinite net event source',lambda:J.combine(a,b,state,state))
a=row();b=copy.deepcopy(a);b['forward']['number'].update(value=None,representable=False)
rejection('gross positive underflow flag',lambda:J.combine(a,b,state,state))
rejection('same sign different radiation affinity',lambda:J.compatible(state,state,scenario('FD',eq-.8*2),scenario('FD',eq-.4*2)))
wrong=copy.deepcopy(state);wrong['T']=2.1
rejection('changed exact material temperature',lambda:J.compatible(state,wrong,scenario(),scenario()))
# Reproduce missing hash coverage with a private verbatim code copy and
# deliberately synthetic accepted-status manifest. No real source files change.
base=H/'synthetic_fixture';dest=base/'antinu_source_join';dest.mkdir(parents=True,exist_ok=True)
(dest/'join.py').write_bytes((R/'antinu_source_join/join.py').read_bytes())
(dest/'ACCEPTED_MANIFEST.json').write_text(json.dumps(dict(status='accepted_selected_sources',sha256={},synthetic_test_only=True)))
for name in ['hot_beta_moments','positron_capture_moments']:(base/name).mkdir(exist_ok=True)
hs=dict(empty=row(),LTE=row('FD'),emission_dominated=row('FD'));hs['emission_dominated']['scenario']['mu_nu']=eq-.8*2
ps=dict(empty=row(),LTE=row('FD'),noneq=row('FD'));ps['noneq']['scenario']['mu_nu']=eq-.4*2
(base/'hot_beta_moments/results.json').write_text(json.dumps(dict(all_pass=True,rows=[dict(matched_state=state),dict(result=dict(controls=dict(rest_order=32),scenarios=hs))])))
(base/'positron_capture_moments/refinement.json').write_text(json.dumps(dict(passed=True,rows=[dict(kind='total',result=dict(controls=dict(initial_order=12),scenarios=ps))])))
(base/'positron_capture_moments/MATCHED_STATE.json').write_text(json.dumps(state))
spec=importlib.util.spec_from_file_location('synthetic_join_copy',dest/'join.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
rejection('accepted status with empty source and review hash coverage',S.build)
(H/'initial_results.json').write_text(json.dumps(dict(tested_source_sha256=hashlib.sha256((R/'antinu_source_join/join.py').read_bytes()).hexdigest(),checks=checks,witnesses=witnesses,synthetic_only=True),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c['name']for c in checks if not c['passed']])))
