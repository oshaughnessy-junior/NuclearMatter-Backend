"""Post-guard exact-state replay; not a new physical-source experiment."""
from pathlib import Path
import json,time
import numpy as np
import box as b
P=Path(__file__).resolve().parent;old=json.loads((P/'initial83pass_before_rate_guards.json').read_text());g=b.grid(old['grid']['edges']);nB=old['nB'];checks=[];start=time.monotonic()
for run in old['runs']:
 for k,row in enumerate(run['rows']):
  before=row['before'];T,Y=before['T'],before['Ye'];r=next(x for x in old['physical_rate_evaluations']if x['T']==T and x['Ye']==Y)
  r={**r,'j':np.array(r['j']),'chi':np.array(r['chi']),'edges':np.array(r['edges'])}
  t,y,f,_=b.step(T,Y,np.array(before['f']),nB,g,row['exchange']['dt'],r)
  errors=[abs(t-row['after']['T']),abs(y-row['after']['Ye']),float(np.max(abs(f-np.array(row['after']['f']))))]
  checks.append(dict(name=f'{run["steps"]} step{k} guarded exact state replay',errors=errors,passed=max(errors)<1e-13))
report=dict(passed=all(c['passed']for c in checks),checks=checks,runtime_seconds=time.monotonic()-start,scope='Each preserved physical coefficient is replayed only at its exact original state/grid; guards change no valid output. Not a fresh trajectory/source calculation.')
(P/'guard_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(len(checks),report['passed'],report['runtime_seconds'])
