from pathlib import Path
import sys, json, dataclasses, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
index, variation = int(sys.argv[1]), int(sys.argv[2])
spec = C.LOCK['small_inner_controls'][index]
change = spec['variations'][variation]
controls = dataclasses.replace(C.FINE, **change['changes'])
path = C.ROOT/'controls'/f'control_{index:02d}_{variation}.json'
if path.exists(): raise RuntimeError('Refuse overwriting actual control evidence')
out = dict(**C.metadata(), channel=spec['channel'], energy_MeV=spec['E'], variation=change, controls=dataclasses.asdict(controls), complete=False)
start = time.monotonic()
try:
    C.preflight()
    row = C.W.spectral_coefficients(spec['channel'], C.BATH, spec['E'], controls)
    row.__post_init__(); C.verify()
    if row.j.log is None or row.true_chi.log is None: raise ArithmeticError('Positive integrated branches unresolved')
    out.update(result=dataclasses.asdict(row), KMS_log_residual=row.kms_log_residual, complete=True)
except Exception as error:
    out['error'] = dict(type=type(error).__name__, message=str(error))
out['seconds'] = time.monotonic()-start
path.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print(index, change['name'], spec['channel'], spec['E'], 'COMPLETE' if out['complete'] else out['error'], out['seconds'], flush=True)
