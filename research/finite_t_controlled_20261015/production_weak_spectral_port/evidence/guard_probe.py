from pathlib import Path
import sys,json,dataclasses
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
b=W.FreeElectronBath(**json.loads((R/'LOCK.json').read_text())['bath']);rule=W.EnergyRule.gauss(2.,3.,2);ctrl=W.SpectralControls();rows=[]
for name,fn in [('FD malformed bath',lambda:W.reconstruct_spectral_moments('EC',object(),W.Radiation.fd(2.,0.,0.),rule,ctrl)),('empty malformed bath',lambda:W.reconstruct_spectral_moments('EC',object(),W.Radiation.empty(),rule,ctrl))]:
 try:fn();kind='accepted';message=''
 except Exception as err:kind=type(err).__name__;message=str(err)
 rows.append(dict(name=name,expected='ValueError',actual=kind,message=message,passed=kind=='ValueError'))
(R/'evidence/guard_probe_initial.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
