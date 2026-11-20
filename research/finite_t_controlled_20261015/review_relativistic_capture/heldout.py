"""Separate bounded physical integrations at one new matched free EOS state."""
from pathlib import Path
import sys,json,math,time,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_capture_integrals'));import capture
sys.path.insert(0,str(ROOT/'relativistic_capture_oracle'));import events
paths=[p for folder in ('relativistic_capture_integrals','relativistic_capture_oracle')for p in (ROOT/folder).iterdir()if p.is_file()]+[ROOT/'relativistic_free_eos/free_baryons.py',ROOT/'relativistic_tensor/tensor.py',ROOT/'thermodynamics/lepton_reference.py',ROOT/'muon_full_current/current.py',ROOT/'muon_vacuum/vacuum_cc.py']
hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
mode=sys.argv[1];before=hashes();start=time.monotonic();m=capture.EOS.matter_state(2.7,1.2e-4,.26,.035);bath=events.Bath(m.temperature,m.neutron.mass,m.proton.mass,m.neutron.chemical_potential,m.proton.chemical_potential,m.muon.chemical_potential,157.,m.muon.mass)
if mode in('capture12','capture16'):
 order=12 if mode=='capture12'else 16;r=capture.integrate(m,157.,lepton_order=order,angle_order=order,energy_order=2*order,phi_order=4,tail=60. if order==12 else 80.,outer_tail=40. if order==12 else 56.)
else:
 if mode not in('events10','events14'):raise ValueError('Expected named bounded mode')
 order=10 if mode=='events10'else 14
 def checkpoint(value):
  (P/f'{mode}_partial.json').write_text(json.dumps(dict(mode=mode,partial=value,complete=False,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
  if time.monotonic()-start>105:raise TimeoutError('Heldout event105s guard')
 r=events.integrate(bath,radial=order,incident=order,outgoing=16 if order==10 else 24,azimuth=8 if order==10 else 12,tail=60. if order==10 else 80.,checkpoint=checkpoint)
if hashes()!=before:raise RuntimeError('Frozen source/data changed during comparison')
(P/f'{mode}.json').write_text(json.dumps(dict(mode=mode,result=r,matter=m.asdict(),bath=vars(bath),source_hashes_before=before,source_hashes_after=hashes(),runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(mode,'seconds',time.monotonic()-start,'chi',r['chi'],'j',r['j'])
