"""Recover only complete accumulated sums from last completed panel checkpoint."""
from pathlib import Path
import json,math
p=Path(__file__).resolve().parent;d=json.loads((p/'refinement_partial.json').read_text());v=d['partial'];assert v['panel']==v['panels']-1
old=[r for r in json.loads((p/'results.json').read_text())['rows']if r['label']=='warm'][-1]['result'];checks=[]
for name in ['chi','j']:
 err=math.expm1(old[name]['log']-v[name]['log']);checks.append(dict(name=name+'14→20 warm refinement',error=err,tolerance=3e-6,passed=abs(err)<3e-6))
err=v['density']/6.999999999999623e-5-1;checks.append(dict(name='fine warm forward density',error=err,tolerance=3e-9,passed=abs(err)<3e-9))
T=2.;Enu=150.;mul=97.74668143468162;mub=930.6761152463529;mua=933.6807508996585;err=v['j']['log']-v['chi']['log']+(Enu-(mul+mub-mua))/T;checks.append(dict(name='fine separately integrated inverse KMS',error=err,tolerance=3e-10,passed=abs(err)<3e-10))
(p/'refinement.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,result=v,controls=dict(radial=20,incident=20,outgoing=32,azimuth=16,tail=80),provenance='Recovered complete positive sums from final panel4of5 checkpoint after JSON numpy-bool reporting failure. No fine shell-residual or event-count result retained or claimed.',runtime_seconds=d['runtime_seconds']),indent=2)+'\n');print(checks)
