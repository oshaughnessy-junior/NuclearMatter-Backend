"""Frozen forward agreement and exact EOS/current/checkpoint provenance gates."""
from pathlib import Path
import sys,json,math,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_free_eos'));import free_baryons as eos
checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
a=json.loads((ROOT/'relativistic_capture_integrals/integration_results.json').read_text());extra=json.loads((ROOT/'relativistic_capture_integrals/heldout_controls_results.json').read_text());b=json.loads((ROOT/'relativistic_capture_oracle/results.json').read_text());fine=json.loads((ROOT/'relativistic_capture_oracle/refinement.json').read_text());partial=json.loads((ROOT/'relativistic_capture_oracle/refinement_partial.json').read_text())
for label,density in [('warm',1e-4),('dilute',1e-15)]:
 x=max([r for r in a+extra if r['state']['nB']==density],key=lambda r:r['controls']['lepton_order']);old=[r for r in b['rows']if r['label']==label][-1];y=fine['result']if label=='warm'else old['result'];state=x['state'];m=eos.matter_state(state['temperature'],state['nB'],state['Ye'],state['Ymu']);bath=old['bath']
 gate(label+' exact fresh neutron mass',abs(m.neutron.mass-bath['ma']),1e-12);gate(label+' exact fresh proton mass',abs(m.proton.mass-bath['mb']),1e-12);gate(label+' exact muon mass',abs(m.muon.mass-bath['ml']),1e-12)
 for species,key in [('neutron','mua'),('proton','mub'),('muon','mul')]:gate(label+' fresh totalmu '+species,abs(getattr(m,species).chemical_potential-bath[key]),2e-10)
 gate(label+' current bare mass',abs(bath['Mweak']-(m.neutron.mass+m.proton.mass)/2),1e-12)
 for k in ('j','chi'):gate(label+' exact frozenforward '+k,abs(math.expm1(x[k]['log']-y[k]['log'])),3e-6)
 gate(label+' actual independently integrated EOS density',abs(y['density']/m.neutron.density-1),3e-9)
 expected=-(x['Enu']-(m.muon.chemical_potential+m.proton.chemical_potential-m.neutron.chemical_potential))/m.temperature
 gate(label+' author separateinverse KMS',abs(x['j']['log']-x['chi']['log']-expected),3e-10);gate(label+' oracle separateinverse KMS',abs(y['j']['log']-y['chi']['log']-expected),3e-10)
 for which,v in [('author',x),('oracle',y)]:
  if label=='dilute':gate(which+' cold inverse finite log notordinaryzero',float(not math.isfinite(v['j']['log'])or v['j']['value']is not None or v['j']['representable']),.1)
 rows.append(dict(label=label,author=x,oracle=y,oracle_bath=bath,fresh_matter=m.asdict()))
gate('recoveredfine finalpanel actuallycomplete',float(partial['partial']['panel']!=partial['partial']['panels']-1),.1)
for key in ('chi','j'):gate('recoveredfineexact '+key,abs(partial['partial'][key]['log']-fine['result'][key]['log']),1e-15)
gate('unavailablefine shellmetadata notinvented',float('max_fourmomentum_residual'in fine['result']),.1)
# Explicit source-independent physical current parameters, not fitted factors.
for row in rows:
 f=row['author']['form_factors'];gate(row['label']+' current gv',abs(f['gv']-1.),1e-15);gate(row['label']+' current ga',abs(f['ga']-1.2723),1e-15);gate(row['label']+' current F2',abs(f['xi']-3.706),1e-15)
(P/'frozen_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks))
for c in checks:
 if not c['passed']:print('FAIL',c)
