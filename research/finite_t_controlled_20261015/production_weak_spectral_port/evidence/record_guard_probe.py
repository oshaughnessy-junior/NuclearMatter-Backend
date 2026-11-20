from pathlib import Path
import sys,json,dataclasses
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
# Saved actual pilot data are only used to form constructor arguments for the
# structural admission adversary, not as physical feedback or fresh rates.
x=json.loads((R/'evidence/pilot_EC_2_empty.json').read_text())['result']
b=W.FreeElectronBath(**x['bath']);c=W.SpectralControls(**x['controls']);rad=W.Radiation.empty();rule=W.EnergyRule(tuple(x['rule']['nodes']),tuple(x['rule']['weights']),x['rule']['lower'],x['rule']['upper'])
args=dict(channel='EC',bath=b,radiation=rad,rule=rule,controls=c,nodes=(),forward=(),backward=(),net=(),entropy_production=None)
rows=[]
for name,kw in [('missing all nodes',{}),('forged whole-moment qualification',dict(whole_moment_qualified=True))]:
 try:W.ReconstructionResult(**args,**kw);kind='accepted'
 except Exception as err:kind=type(err).__name__
 rows.append(dict(name=name,expected='ValueError',actual=kind,passed=kind=='ValueError'))
(R/'evidence/record_guard_initial.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
