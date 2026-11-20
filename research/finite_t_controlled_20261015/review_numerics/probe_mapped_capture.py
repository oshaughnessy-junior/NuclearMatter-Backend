"""Independent held-out threshold checks for endpoint-mapped capture."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'capture_endpoint'))
import mapped_capture as m
import allowed_capture as a
from free_density_response import PairState,response
c=a.constants
pair=PairState(3.,939.,c.Q,0.,c.Q+20.,5.)
s=a.OpticalState(pair.reverse(),-5.)
def independent(Enu,nangle):
    cs,ws=leggauss(nangle)
    def f(p):
        Ee=math.hypot(p,c.ME)
        angular=sum(w*a.angular_factor(Ee,z)*response(s.baryons,math.sqrt(Enu**2+p*p-2*Enu*p*z),Enu-Ee) for z,w in zip(cs,ws))
        return p*p*expit((Ee-s.mu_lepton)/s.baryons.T)*angular
    es=[c.ME,.53,.6,.8,1.,1.3,2.,3.,5.,10.,25.,60.]
    ps=[math.sqrt(E*E-c.ME*c.ME) for E in es]
    return a.K*sum(quad(f,lo,hi,epsabs=1e-30,epsrel=2e-10,limit=200)[0] for lo,hi in zip(ps[:-1],ps[1:]))
checks=[];samples=[]
def check(name,error,gate):checks.append(dict(name=name,error=float(error),passed=bool(error<gate)))
for E in [1.79,1.8,1.81]:
    ref=independent(E,128)
    logs=[m.log_coefficients(s,E,energy_order=k,angle_order=k) for k in [48,96]]
    check(f'independent momentum normalization E{E}',abs(math.exp(logs[0][1])/ref-1),3e-8)
    check(f'mapped48to96 refinement E{E}',abs(math.expm1(logs[0][1]-logs[1][1])),3e-8)
    check(f'KMS E{E}',abs(logs[0][0]-logs[0][1]+(E-s.mu_neutrino)/s.baryons.T),3e-11)
    samples.append(dict(E=E,independent_momentum_chi=ref,mapped_log_coefficients=logs))
out=dict(passed=all(x['passed'] for x in checks),checks=checks,samples=samples,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(m.__file__),Path(a.__file__)]})
Path(__file__).with_name('mapped_capture_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']]),indent=2))
