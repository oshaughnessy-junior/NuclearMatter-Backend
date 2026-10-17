from pathlib import Path
import json,math,time
import resolved as r
start=time.monotonic();p=Path(__file__).resolve().parent;initial=json.loads((p/'results.json').read_text())['rows'][0]
s=r.model.make_state(initial['reference'],.01).optical('nue');ref=initial['methods'][1];refg=initial['methods'][3]
runs=[r.adaptive_yq(s,.03,order=32,tail=160.),r.adaptive_yq(s,.03,order=64,tail=100.),r.gauss_Ec(s,.03,order=16,angle_order=48,tail=160.),r.gauss_Ec(s,.03,order=24,angle_order=32,tail=160.),r.gauss_Ec(s,.03,order=24,angle_order=48,tail=100.)];checks=[]
for i,run in enumerate(runs):
 reference=ref if i<2 else refg
 for key in ['log_j','log_chi']:
  error=abs(math.expm1(run[key]-reference[key]));checks.append(dict(name=f'individual control {i} {key}',error=error,tolerance=3e-6,passed=error<3e-6))
result=dict(passed=all(c['passed']for c in checks),checks=checks,runs=runs,runtime_seconds=time.monotonic()-start)
(p/'refinement.json').write_text(json.dumps(result,indent=2)+'\n');print(len(checks),result['passed'],result['runtime_seconds'])
