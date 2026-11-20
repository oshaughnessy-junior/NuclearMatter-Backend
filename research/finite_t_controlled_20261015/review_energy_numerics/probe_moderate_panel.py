"""Bounded independent log-energy check of one repaired degenerate bulk panel."""
from pathlib import Path
import sys,json,math,time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantum_slab'))
import fixed_bath as b
start=time.monotonic();matter=b.matter_state(2.,.002,.001)
mu,mw=leggauss(80);mu=.5*(mu+1);mw=.5*mw
def run(order):
    xs,ws=leggauss(order);out=np.zeros(4)
    zlo,zhi=math.log1p(40.),math.log1p(70.)
    for x,w in zip(xs,ws):
        if time.monotonic()-start>15:raise TimeoutError('15second supplementary experiment limit')
        z=.5*(zlo+zhi)+.5*(zhi-zlo)*x;E=math.expm1(z);weight=.5*(zhi-zlo)*w*math.exp(z)
        c=b.coefficients(matter,'nue',E,order=48,tail=40.);left,right=b.boundary_occupations(E)
        escape=-np.expm1(-c['rate_s']*1e5/(b.LIGHT*mu));eq=c['j_s']/c['rate_s']
        A=b.LIGHT/(2*math.pi**2*197.3269804**3)*E*E*1e39
        en=A/b.LIGHT*c['j_s'];net=A*.5*float(np.dot(mw,mu*((eq-left)*escape+(eq-right)*escape)))
        out+=weight*np.array([en,E*en,net,E*net])
    return out
coarse,fine=run(20),run(28)
source=json.loads((ROOT/'quantum_energy_moments/moderate_refined_results.json').read_text())
p=next(p for p in source['panel_refinements']if p['lo']==40.)['fine48']
reference=np.array([p[0],p[1],p[6]-p[7],p[12]-p[13]])
checks=[]
for i,name in enumerate(['volume_number','volume_energy','slab_number_exchange','slab_energy_exchange']):
    for label,value,ref in [('independent reference',fine[i],reference[i]),('log-grid refinement',coarse[i],fine[i])]:
        error=abs(value/ref-1);checks.append(dict(name=name+' '+label,error=float(error),passed=bool(error<3e-5)))
result=dict(passed=all(c['passed']for c in checks),checks=checks,coarse=coarse.tolist(),fine=fine.tolist(),reference=reference.tolist(),runtime_seconds=time.monotonic()-start)
Path(__file__).with_name('moderate_panel_refined_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
