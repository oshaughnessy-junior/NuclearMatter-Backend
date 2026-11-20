from exchange import *
import json
P=Path(__file__).resolve().parent;checks=[]
for label,call in [('Tnan',lambda:Radiation(math.nan,0)),('Tzero',lambda:Radiation(0.,0)),('munan',lambda:Radiation(1.,math.nan)),('packetint',lambda:Radiation(1.,0.,1)),('cutoffnan',lambda:control(math.nan,8,16,24)),('cutoffzero',lambda:control(0.,8,16,24)),('orderbool',lambda:control(72.,True,16,24)),('orderfloat',lambda:control(72.,8,16.,24)),('ordersmall',lambda:control(72.,8,2,24)),('radenergynegative',lambda:Radiation(2.,3.).logs(-1)),('radenergynan',lambda:Radiation(2.,3.).logs(math.nan))]:
 try:call();reject=False
 except ValueError:reject=True
 checks.append(dict(name=label,passed=reject))
(P/'controls.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
