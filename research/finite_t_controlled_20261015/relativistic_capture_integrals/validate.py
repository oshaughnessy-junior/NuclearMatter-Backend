"""Bounded report checks; no rate recalculation or coarse-failure suppression."""
from __future__ import annotations
from pathlib import Path
import json,math,hashlib,platform
import numpy as np,scipy
P=Path(__file__).resolve().parent
rows=json.loads((P/'integration_results.json').read_text());anchors=json.loads((P/'anchor_results.json').read_text());extra=json.loads((P/'heldout_controls_results.json').read_text())
checks=[]
def check(name,value,limit):checks.append(dict(name=name,error=abs(float(value)),limit=limit,passed=abs(value)<=limit))
warm=[r for r in rows if r['label']=='physical'];cold=[r for r in rows if r['label']=='dilute'];truth=anchors[-1]['chi_s']
for r in rows+extra:
 check(f'KMS_{r.get("label",r.get("probe"))}_{r["controls"]["lepton_order"]}',r['kms_log_actual']-r['kms_log_expected'],2e-11)
 for name in('j','chi'):
  assert math.isfinite(r[name]['log'])
  if r[name]['representable']:check('ordinary_log_'+name,math.log(r[name]['value'])-r[name]['log'],2e-12)
check('warm_final_joint_refinement',warm[-2]['chi']['value']/warm[-1]['chi']['value']-1,1e-7)
for r in extra:
 if r['probe']!='dilute20':check('warm_control_'+r['probe'],r['chi']['value']/warm[-1]['chi']['value']-1,1e-7)
for i,a in enumerate(anchors):check('anchor_density_'+str(i),a['density_ratio']-1,2e-10)
check('anchor_refinement',anchors[-2]['chi_s']/truth-1,1e-8)
check('dilute16_absolute',cold[-1]['chi']['value']/truth-1,3e-6)
fine=next(x for x in extra if x['probe']=='dilute20')
check('dilute20_absolute',fine['chi']['value']/truth-1,3e-6)
check('dilute16_20_refinement',fine['chi']['value']/cold[-1]['chi']['value']-1,3e-6)
assert fine['j']['value']is None and fine['j']['reason']=='underflow'and fine['j']['log']< -745
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),retained_coarse_absolute_failures=[dict(order=r['controls']['lepton_order'],relative_error=r['chi']['value']/truth-1,gate=3e-6,qualified=abs(r['chi']['value']/truth-1)<=3e-6)for r in cold],qualification='Selected capture points only; no crossed beta inventory or GM1; upstream review required')
(P/'validation_results.json').write_text(json.dumps(out,indent=2))
root=P.parent;paths=[P/'capture.py',P/'dilute_anchor.py',P/'validate.py',P/'run_study.py',P/'RESEARCH_NOTE.md',root/'relativistic_tensor/tensor.py',root/'muon_full_current/current.py',root/'muon_vacuum/vacuum_cc.py',root/'relativistic_free_eos/free_baryons.py',root/'thermodynamics/lepton_reference.py']
manifest=dict(source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),raw_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in P.glob('*.json')if p.name!='CHECKPOINT.json'})
(P/'CHECKPOINT.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(out,indent=2))
if out['failed']:raise SystemExit(1)
