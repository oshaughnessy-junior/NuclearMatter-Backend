from pathlib import Path
import sys,json,math,time
import numpy as np
sys.dont_write_bytecode=True
from oracle import *
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'response_reference'))
from free_density_response import PairState,response,HBARC
checks=[];rows=[];start=time.monotonic()
for M in [1e5,1e6,1e7]:
 p=Point(2,M,M,M-10,M-11,10,10.0001,.2,.51099895069,1.,False,M)
 r=integrate(p,32,8,50);pl=math.sqrt(p.El*p.El-p.ml*p.ml);s=response(PairState(p.T,M,M,M,p.mua,p.mub),r['q'],r['w'])
 angular=(1+1.2723**2)+(1-1.2723**2)*(pl/p.El)*p.cosine
 chi=GF**2*VUD**2*HBARC**3/(4*math.pi**2*HBAR)*p.El*pl*angular*s*expit((p.El-p.mul)/p.T)
 error=r['chi']/chi-1;rows.append(dict(mass=M,relativistic=r['chi'],NR_allowed=chi,relative_error=error))
 checks.append(dict(name=f'M{M} heavy-mass absolute allowed limit',error=error,tolerance=2e-4 if M==1e5 else 3e-5 if M==1e6 else 1e-5,passed=abs(error)<(2e-4 if M==1e5 else 3e-5 if M==1e6 else 1e-5)))
checks.append(dict(name='approach to NR limit',passed=abs(rows[-1]['relative_error'])<abs(rows[0]['relative_error'])/20))
Path(__file__).with_name('nr_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2,default=lambda x:x.item())+'\n');print(rows)
