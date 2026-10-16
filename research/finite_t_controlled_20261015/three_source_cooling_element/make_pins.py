"""One-time local pin record, never update after evidence begins."""
from pathlib import Path
import json,hashlib,sys,platform
P=Path(__file__).parent;R=P.parent
paths=list(json.loads((R/'transparent_capture_balance/UPSTREAM.json').read_text()))+['transparent_capture_balance/balance.py','transparent_capture_balance/UPSTREAM.json','transparent_capture_balance/EC_REVIEW_ACCEPTANCE.json','hot_electron_capture_moments/electron_capture.py','review_hot_electron_capture/REVIEW.md','review_hot_electron_capture/CHECKPOINT.json','review_transparent_capture_balance/REVIEW.md','review_transparent_capture_balance/CHECKPOINT.json','three_source_cooling_element/worker.py','three_source_cooling_element/client.py']
expected=['three_source_cooling_element/worker.py','transparent_capture_balance/balance.py','hot_electron_capture_moments/electron_capture.py','hot_beta_moments/hot_moments.py','review_pc_moments/reference.py','relativistic_capture_oracle/events.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py','relativistic_crossed_contract/beta.py','relativistic_crossed_thermal/kernel.py','relativistic_tensor/tensor.py','muon_full_current/current.py','muon_vacuum/vacuum_cc.py']
if(P/'PINS.json').exists():raise RuntimeError('Refusing pin rewrite')
x=dict(files={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in sorted(set(paths))},expected_loaded_python=expected,accepted_balance=True,accepted_balance_scope='selected T2 source balance; new cooling/changeT qualification pending',environment=dict(python=sys.version,platform=platform.platform(),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
(P/'PINS.json').write_text(json.dumps(x,indent=2)+'\n');print('Pinned',len(x['files']),'files and',len(expected),'loaded Python paths')
