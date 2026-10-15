"""Harvest only fully completed independent12/cos/El integral; retain budget stop."""
from pathlib import Path
import json,math
p=Path(__file__).resolve().parent;d=json.loads((p/'results.json').read_text());completed=[r for r in d['rows']if 'angular_order'in r];assert len(completed)==1;r=completed[0];assert r['angular_order']==12 and r['upper']==600.;checks=[]
for name,target in [('chi',958899.1195336229),('j',745802384.5583102)]:
 err=r[name]/target-1;checks.append(dict(name='independent adaptive direct-El vs frozen electron '+name,error=err,tolerance=1e-6,passed=abs(err)<1e-6))
T=8.;E=40.;mul=203.11989906077872;mua=1000.1139607177219;mub=890.2455022016085;err=math.log(r['j']/r['chi'])+(E-(mul+mub-mua))/T;checks.append(dict(name='independently accumulated coefficients KMS',error=err,tolerance=3e-12,passed=abs(err)<3e-12))
(p/'selected_comparison.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,completed_integral=r,provenance='One COMPLETE independent adaptive DIRECT-El integral at12angular; attempted20angular/900MeV refinement was interrupted by108s experiment guard and remains partial. No20order rate or independent tail/refinement result claimed.'),indent=2)+'\n');print(checks)
(p/'budget_stop.json').write_text(json.dumps(dict(error='TimeoutError: Bounded independent outer quadrature',runtime_seconds=d['runtime_seconds'],phase=d['phase'],completed_integral_count=1,partial_state_preserved=True,numerical_gate_failure=False),indent=2)+'\n')
