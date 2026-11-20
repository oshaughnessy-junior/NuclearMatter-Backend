from pathlib import Path
import sys,json,dataclasses
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_current as C
from nmbackend.reference import _weak_spectral_capture as S
import numpy as np
from numpy.polynomial.legendre import leggauss
lock=json.loads((R/'LOCK.json').read_text());b=W.FreeElectronBath(**lock['bath']);rows=[]
points=lock['new_differential']+[['PC',1.7,.5987748936352157,-.1995027501115606]]
for ch,E,el,c in points:
 r=W.capture_differential_coefficients(ch,b,E,el,c,W.SpectralControls.fine())
 rows.append(dict(point=[ch,E,el,c],result=dataclasses.asdict(r),KMS=r.kms_log_residual))
(R/'evidence/differential.json').write_text(json.dumps(rows,indent=2)+'\n');print([(r['point'],r['result']['j']['value'],r['result']['true_chi']['value'])for r in rows])
