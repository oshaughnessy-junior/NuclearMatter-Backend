"""Owned validation driver; never imported by runtime; exact mirror source origin."""
from pathlib import Path
import sys,time,json,dataclasses,math
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
assert Path(W.__file__).resolve()==(R/'test_mirror/src/nmbackend/reference/weak_reference.py').resolve()
fixture=json.loads((R/'staged/tests/data/free_weak_spectral_reference.json').read_text())
b=W.FreeElectronBath(**fixture['spectra'][0]['bath'])
name=sys.argv[1];cname=sys.argv[2];E=float(sys.argv[3]);profile=sys.argv[4]
ctl=W.SpectralControls() if profile=='base' else W.SpectralControls.fine()
start=time.monotonic()
try:
 row=W.spectral_coefficients(cname,b,E,ctl)
 out=dict(name=name,result=dataclasses.asdict(row),KMS=row.kms_log_residual,relaxation=dataclasses.asdict(row.relaxation),seconds=time.monotonic()-start,source=W.__file__)
except Exception as error:
 out=dict(name=name,error=type(error).__name__,message=str(error),seconds=time.monotonic()-start)
(R/'evidence'/f'{name}.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out)[:1500])
if 'error'in out:sys.exit(1)
