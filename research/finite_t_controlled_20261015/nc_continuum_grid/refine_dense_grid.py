"""Resolve coupled angular/energy aliasing rather than infer it from moments."""
from pathlib import Path
import json,time
import numpy as np
import redistribution as r
P=Path(__file__).resolve().parent;initial=json.loads((P/'results.json').read_text());rows=[];checks=[];start=time.monotonic()
for old in initial['rows']:
 s=r.nc.state_density(old['state']['T'],old['state']['density'],old['state']['offset']);E=old['Ein'];rad=r.Radiation(**old['radiation']);ref=np.array(old['references'][1]['values']);row=dict(label=old['label'],grids=[]);rows.append(row)
 for order,angle in [(32,192),(48,192),(48,384)]:
  value=r.energy_grid(s,E,rad,order=order,angle_order=angle,tail=70.);error=abs(np.array(value['values'])/ref-1);value['relative_errors']=error.tolist();row['grids'].append(value)
  for k,e in enumerate(error):checks.append(dict(name=f'{old["label"]} outer{order}q{angle} {r.LABELS[k]}',error=float(e),tolerance=3e-5,passed=bool(e<3e-5)))
  (P/'dense_grid_refinement.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
for row in rows:print(row['label'],[(x['order'],x['angle_order'],max(x['relative_errors']))for x in row['grids']])
