from pathlib import Path
import json,time,math
import numpy as np
import log_gain as r
P=Path(__file__).resolve().parent;initial=json.loads((P/'results.json').read_text());row=next(x for x in initial['rows']if x['label']=='cold_repair');sp=r.nc.state_density(.08,.0042,0.);E=19.6;rad=r.Radiation('FD',.4,5.);cut=r.original.upper(sp,E,70.);runs=[];checks=[];start=time.monotonic()
for eo,ao,mode,factor in [(128,384,'kms',1.),(160,512,'kms',1.),(192,640,'kms',1.),(192,640,'direct',1.),(192,640,'kms',1.5)]:
 if time.monotonic()-start>105:raise TimeoutError('Refine105sbound')
 runs.append(r.integrate(sp,E,rad,energy_order=eo,angle_order=ao,cutoff=cut*factor,reverse=mode))
 (P/'cold_refinement.json').write_text(json.dumps(dict(runs=runs,checks=checks,runtime_seconds=time.monotonic()-start,phase='partial'),indent=2)+'\n')
ref=np.array(runs[2]['values']);arch=np.array(row['archived_reference'])
for j in [0,1,3,4]:
 for k,e in enumerate(abs(np.array(runs[j]['values'])/ref-1)):checks.append(dict(name=f'coldrefine{j} '+r.original.LABELS[k],error=float(e),tolerance=3e-5 if j!=4 else1e-8,passed=bool(e<(3e-5 if j!=4 else1e-8))))
for k,e in enumerate(abs(ref/arch-1)):checks.append(dict(name='cold archive q-outer reference '+r.original.LABELS[k],error=float(e),tolerance=3e-5,passed=bool(e<3e-5)))
result=dict(passed=all(c['passed']for c in checks),phase='complete',runs=runs,checks=checks,runtime_seconds=time.monotonic()-start,scope='Exactcoldselectedpoint; originalcoarsegainfailuresretainFAIL')
(P/'cold_refinement.json').write_text(json.dumps(result,indent=2)+'\n');print(len(checks),result['passed'],result['runtime_seconds']);print([c for c in checks if not c['passed']])
