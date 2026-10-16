from pathlib import Path
import sys,json,dataclasses,math,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
b=W.FreeElectronBath(**json.loads((R/'LOCK.json').read_text())['bath']);fine=W.SpectralControls.fine();rows=[]
for ch,E,changes in [('EC',7.3,[dict(tail=90.),dict(outer_tail=50.),dict(phi_order=10),dict(lepton_order=16),dict(angle_order=24),dict(energy_order=24)]),('NEUTRON_DECAY',.83,[dict(tail=90.),dict(parent_order=16),dict(cone_order=20),dict(electron_order=20),dict(decay_phi_order=14)])]:
 reference=json.loads((R/'evidence'/f'new_{ch}_{E}_fine.json').read_text())['result']
 for change in changes:
  start=time.monotonic();r=W.spectral_coefficients(ch,b,E,dataclasses.replace(fine,**change));errors=[abs(math.expm1(getattr(r,k).log-reference[k]['log']))for k in ('j','true_chi')]
  row=dict(channel=ch,E=E,change=change,logs=[r.j.log,r.true_chi.log],errors=errors,gate=3e-6,passed=max(errors)<3e-6,seconds=time.monotonic()-start);rows.append(row)
  (R/'evidence/isolated_controls_partial.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
(R/'evidence/isolated_controls.json').write_text(json.dumps(rows,indent=2)+'\n')
