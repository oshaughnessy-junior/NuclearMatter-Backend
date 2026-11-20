from pathlib import Path
import sys,json,dataclasses
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S
rows=[]
for p in sorted((R/'evidence').glob('pilot_*.json')):
 data=json.loads(p.read_text())['result'];b=W.FreeElectronBath(**data['bath']);c=W.SpectralControls(**data['controls']);rad=W.Radiation(**data['radiation']);rule=W.EnergyRule(tuple(data['rule']['nodes']),tuple(data['rule']['weights']),data['rule']['lower'],data['rule']['upper'])
 nodes=[]
 for n in data['nodes']:
  n=dict(n);n.update(bath=b,controls=c,j=W.SpectralQuantity(**n['j']),true_chi=W.SpectralQuantity(**n['true_chi']),geometry=tuple(n['geometry']))
  nodes.append(W.SpectralResult(**n))
 ent=data['entropy_production'];ent=None if ent is None else W.SignedQuantity(**ent) if 'sign'in ent else W.PositiveQuantity(**ent)
 q=dict(data);q.update(bath=b,radiation=rad,rule=rule,controls=c,nodes=tuple(nodes),forward=tuple(W.PositiveQuantity(**v)for v in data['forward']),backward=tuple(W.PositiveQuantity(**v)for v in data['backward']),net=tuple(W.SignedQuantity(**v)for v in data['net']),entropy_production=ent,units=tuple(data['units']))
 r=W.ReconstructionResult(**q)
 identical=json.loads(json.dumps(dataclasses.asdict(r)))==data
 checks=[]
 for patch in [dict(nodes=()),dict(whole_moment_qualified=True),dict(complete_finite_range=False),dict(forward=()),dict(controls=W.SpectralControls.fine()),dict(units=('bad','bad'))]:
  try:dataclasses.replace(r,**patch);checks.append(False)
  except ValueError:checks.append(True)
 rows.append(dict(path=p.name,valid_payload_identical=identical,guard_checks=checks))
(R/'evidence/pilot_record_repair_validation.json').write_text(json.dumps(rows,indent=2)+'\n');print(len(rows),'valid identical',sum(v['valid_payload_identical']for v in rows),'guards',sum(sum(v['guard_checks'])for v in rows))
