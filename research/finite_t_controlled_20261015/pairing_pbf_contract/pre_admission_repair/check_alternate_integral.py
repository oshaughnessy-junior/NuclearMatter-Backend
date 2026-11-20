"""Alternate hyperbolic energy coordinate and fixed angular rules."""
from pathlib import Path
import json,math
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from quasiparticle import log_F_A,log_F_B
P=Path(__file__).resolve().parent

def hyperbolic_A(v,tail):
    # x=v*sinh(t), z=v*cosh(t). Direct alternate measure v^7 cosh(t)^5.
    upper=math.acosh(1+tail/v)
    def f(t):
        c=math.cosh(t);difference=2*math.sinh(t/2)**2
        return c**5*math.exp(-2*v*difference)/(1+math.exp(-v*c))**2
    integral=quad(f,0,upper,epsabs=2e-11,epsrel=2e-12,limit=200)[0]
    return 7*math.log(v)-2*v+math.log(integral)

checks=[];rows=[]
def gate(n,e,tol):checks.append(dict(name=n,error=float(e),tolerance=tol,passed=bool(abs(e)<=tol)))
for v in [.03,1.5,12.,100.]:
    a=hyperbolic_A(v,50.);b=hyperbolic_A(v,70.);r=log_F_A(v)
    gate('hyperbolic coordinate vs scaled x '+str(v),math.expm1(b-r),2e-10)
    gate('hyperbolic tail50/70 '+str(v),math.expm1(a-b),2e-10)
    angles=[]
    for n in [64,96]:
        x,w=leggauss(n);angles.append(-2*v+math.log(sum(ww/2*math.exp(hyperbolic_A(v*math.sqrt(1+3*((xx+1)/2)**2),70.)+2*v) for xx,ww in zip(x,w))))
    gate('fixed64/96 angle '+str(v),math.expm1(angles[0]-angles[1]),2e-10)
    gate('fixed angle vs adaptive angular '+str(v),math.expm1(angles[1]-log_F_B(v)),2e-10)
    rows.append(dict(v=v,hyperbolic_A_log=b,reference_A_log=r,fixed_angular_B_logs=angles,reference_B_log=log_F_B(v)))
o=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,source='Same reduced BCS scalar phase integral with different quasiparticle and angular quadratures; not a new weak-current theory')
(P/'alternate_integral_results.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({k:o[k] for k in ['passed','failed']}))
