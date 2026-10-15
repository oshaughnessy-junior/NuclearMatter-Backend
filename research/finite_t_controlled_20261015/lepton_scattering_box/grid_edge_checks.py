"""Independent forward references at actual eight-node grid edges."""
from pathlib import Path
import json,math
import numpy as np
import box as B
import reference_edge as R
HERE=Path(__file__).resolve().parent;r=json.loads((HERE/'results.json').read_text());g=r['grids'][-1];checks=[];rows=[]
for i,j in[(0,1),(3,4),(6,7)]:
 E,F=g['E'][i],g['E'][j];ref=R.edge(B.T,B.MU,E,F,B.COUPLING.cv,B.COUPLING.ca);actual=g['K'][i][j];err=actual/ref-1;checks.append(dict(name='independent actual grid edge '+str((i,j)),error=err,tolerance=3e-6,passed=abs(err)<=3e-6));rows.append(dict(i=i,j=j,E=E,F=F,physical_stored=actual,independent=ref))
# Time uncertainty in the SAME heat observable used by spectral checks.
for Tr in[1.3,3.]:
 a=[x['run']for x in r['evolutions']if x['N']==8 and x['run']['Tr']==Tr];err=abs(a[1]['matter_heat']/a[2]['matter_heat']-1);checks.append(dict(name=str(Tr)+' time32/64 heat smaller than .1percent',error=err,tolerance=.001,passed=err<=.001))
(HERE/'grid_edge_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],pins=B.SOURCE_PINS),indent=2));print(json.dumps(checks,indent=2))
