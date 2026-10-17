"""New physical antinue/e+ vacuum spinor vs published-invariant anchor."""
from pathlib import Path
import sys,json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
import pc_kernel as pc
sys.path.insert(0,str(pc.R/'relativistic_capture_oracle'));import events as ev
H=Path(__file__).parent;checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
ch=pc.V.Channel(pc.crossed.MP,pc.crossed.MN,pc.ME,True)
ffmodel=pc.K.C.FormFactors()
for E in [2.,5.,20.]:
 xs,ws=leggauss(24);total=0.
 eb=ev.Bath(2.,pc.crossed.MP,pc.crossed.MN,900.,900.,0.,Enu=E,ml=pc.ME,anti=True,Mweak=pc.crossed.MW)
 for c,w in zip(xs,ws):
  p=pc.V.momenta(ch,E,float(c));points=[p[k][None,:]for k in ['k','pa','l','pb']]
  amp=float(ev.amplitude(eb,*points)[0]);q=p['pb']-p['pa'];t=pc.V.dot(q,q);ff=ffmodel.values(t,pc.crossed.MW)
  source=sum(pc.K.C.guo_invariants(ch,p,ff).values())
  gate('literal reciprocal spinors/source '+str(E)+' '+str(float(c)),amp/source-1,3e-10)
  total+=w*amp/(32*math.pi*p['s'])*p['pf']/p['pi']*pc.V.HBARC**2*1e-26
 ref=pc.K.C.total(ch,E,order=24,method='source')
 gate('absolute CM total cross section '+str(E),total/ref-1,3e-11)
 rows.append(dict(Enu=E,full_current_sigma_cm2=total,source_invariant_sigma_cm2=ref))
(H/'vacuum_anchor.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
