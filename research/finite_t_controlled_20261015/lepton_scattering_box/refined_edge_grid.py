"""New fine physical eight-node grid after the retained high-energy edge failure."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.special import expit
import box as B
import reference_edge as R
HERE=Path(__file__).resolve().parent;checks=[];rows=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
g=B.grid(8,24.,angle=40,inner=32,tail=80.)
for i,j in[(0,1),(3,4),(6,7)]:
 E,F=g['E'][i],g['E'][j];v=R.edge(B.T,B.MU,E,F,B.COUPLING.cv,B.COUPLING.ca);gate('fine actual grid edge independent '+str((i,j)),g['K'][i][j]/v-1,3e-6);rows.append(dict(i=i,j=j,E=E,F=F,independent=v,fine=g['K'][i][j]))
E=np.array(g['E']);W=np.array(g['W']);lte=B.collision(g,expit((-.7-E)/B.T));gate('fine actual reverse KMS',g['kms_log_residual'],3e-10);gate('fine actual FDstationarity',max(abs(np.array(lte['dn'])))/lte['activity'],3e-12)
runs=[]
for Tr in[1.3,3.]:
 r=B.evolve(g,Tr,-.7,steps=64);runs.append(r);gate(str(Tr)+' finegrid number ledger',abs(r['number_ledger'])/r['number_initial'],3e-13);gate(str(Tr)+' finegrid energy ledger',abs(r['energy_ledger']/r['matter_heat']),3e-12);gate(str(Tr)+' finegrid entropy increase',min(0.,r['entropy_radiation']+r['entropy_bath'])/r['number_initial'],1e-14)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],grid=g,edge_references=rows,runs=runs,pins=B.SOURCE_PINS,seconds=time.monotonic()-start)
(HERE/'refined_edge_grid_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in ['passed','failed','seconds']},indent=2))
