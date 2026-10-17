"""Actual increased absolute Eout cutoff, even when nominal T-tail is inactive."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.integrate import quad_vec
import redistribution as r
P=Path(__file__).resolve().parent;old=json.loads((P/'results.json').read_text());checks=[];rows=[];start=time.monotonic()
for row in old['rows']:
 s=r.nc.state_density(row['state']['T'],row['state']['density'],row['state']['offset']);E=row['Ein'];rad=r.Radiation(**row['radiation']);reference=row['references'][1];lo=reference['cutoff'];hi=1.5*lo
 value,error=quad_vec(lambda x:r.integrand(s,E,float(x),rad,angle_order=96),lo,hi,epsabs=1e-100,epsrel=1e-8,limit=250)
 relative=abs(value/np.array(reference['values']));pair=r.nc.pair_from_species(s);samples=[]
 for Ep in [lo,(lo+hi)/2,hi]:
  samples.append(dict(Eout=Ep,log_forward_at_cos0=r.nc.log_kernel(pair,E,Ep,0.),log_reverse_at_cos0=r.nc.log_kernel(pair,Ep,E,0.)))
 for k,err in enumerate(relative):checks.append(dict(name=row['label']+' cutoff1.5 '+r.LABELS[k],error=float(err),tolerance=1e-10,passed=bool(err<1e-10)))
 rows.append(dict(label=row['label'],cutoff_before=lo,cutoff_after=hi,tail_moments=value.tolist(),relative_to_reference=relative.tolist(),absolute_quad_error=float(error),sampled_log_kernels=samples,limit='Sampled finite logarithmic tails may be unrepresentable ordinarily; no mathematical infinite-tail bound or relative-tail accuracy claim'))
 (P/'tail_extension.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
