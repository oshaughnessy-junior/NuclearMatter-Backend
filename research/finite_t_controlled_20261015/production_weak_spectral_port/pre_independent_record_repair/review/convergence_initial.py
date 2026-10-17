from pathlib import Path
import json,math,datetime,hashlib
P=Path(__file__).resolve().parent;checks=[]
for ch,E in [('EC',6.431),('PC',4.237),('NEUTRON_DECAY',.337),('NEUTRON_DECAY',.873)]:
 a,b=[json.loads((P/f'pure_{ch}_{E}_{s}.json').read_text())for s in['coarse','fine']]
 for key in['j','chi']:
  er=math.expm1(a['row']['log'+key]-b['row']['log'+key]);checks.append(dict(name=f'{ch}{E}{key} originalcoarse/fine',error=er,gate=3e-6,passed=abs(er)<=3e-6))
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,candidate_imported=False,kind='Originalreferenceaccuracygates remainFAIL; no gatechange')
(P/'INITIAL_CONVERGENCE.json').write_text(json.dumps(out,indent=2)+'\n')
lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='ONEnewhigherquad rung afterpreserved coarsefails; no newphysics or gatechange',original_convergence_sha256=hashlib.sha256((P/'INITIAL_CONVERGENCE.json').read_bytes()).hexdigest(),profiles=dict(capture_high=dict(order=14,qorder=24,inner=40,phi=12,tail=85.,outer_tail=50.),beta_high=dict(parent=18,nu_order=24,electron=24,phi_order=16,tail=85.)),jobs=[dict(channel='EC',E=6.431),dict(channel='PC',E=4.237),dict(channel='NEUTRON_DECAY',E=.873)],gate=3e-6,budget_seconds=105,tail_plan='Afterconvergenceonly isolatecapturematerialtail95atfixedhigh thenoutertail60atfixedhigh; beta.337/.873 fineorhighsameorderstail95. No newpoints/domain.',candidate_imported=False)
(P/'HIGH_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print('INITIAL',out['passed'],'FAIL',len(out['failed']),'HIGHLOCK',hashlib.sha256((P/'HIGH_LOCK.json').read_bytes()).hexdigest())
