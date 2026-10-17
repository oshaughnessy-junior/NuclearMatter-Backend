from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;R=P.parent;h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();A=R/'beta_support';B=R/'beta_review_preparation';b=json.loads((B/'MANIFEST.json').read_text());a=json.loads((A/'MANIFEST.json').read_text())
assert h(B/'CHECKPOINT.json')=='6257b765dbe024b655ed3b9980470187f3c26f82ad125d8f320cafd5dba07dcc'
assert h(A/'MANIFEST.json')=='acec26c884203ed396e7d9ff0684a31568f172826a692b50a52fb589380547f6'
assert all(h(B/f)==s for f,s in b['files_sha256'].items());assert all(h(A/f)==s for f,s in a['files_sha256'].items())
lock={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'blind_prep_files':b['files_sha256'],'blind_prep_checkpoint':h(B/'CHECKPOINT.json'),'author_files':a['files_sha256'],'author_manifest':h(A/'MANIFEST.json'),'jobs':[dict(kind='oracle',E=1.2,parent=16,rest_nu=16,electron=16,phi=8),dict(kind='oracle',E=1.2,parent=24,rest_nu=24,electron=24,phi=12),dict(kind='oracle',E=1.27,parent=24,rest_nu=24,electron=24,phi=12),dict(kind='candidate',E=1.27,parent=24,rest_nu=24,electron=24,phi=12)],'actual_oracle_map':'threshold-shifted parent genLaguerre alpha1.5; rest_nu endpoint gap Jacobi beta.5; affine exact rest electron energy; original energy-cone Jacobian and literal old beta spinor current','branch_relative_gate':3e-6,'max_calls':4,'max_seconds':90,'no_extra_rungs':True}
(P/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(h(P/'LOCK.json'))
