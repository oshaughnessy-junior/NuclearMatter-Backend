"""Read-only old capture endpoint counterexample with momentum integration."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'allowed_opacity'))
import allowed_capture as a
from free_density_response import PairState,response
c=a.constants
pair=PairState(3.,939.,c.Q,0.,c.Q+20.,5.)
s=a.OpticalState(pair.reverse(),-5.)
Enu=1.8
def independent(nangle):
    cs,ws=leggauss(nangle)
    def f(p):
        Ee=math.hypot(p,c.ME)
        angular=sum(w*a.angular_factor(Ee,z)*response(s.baryons,math.sqrt(Enu**2+p*p-2*Enu*p*z),Enu-Ee) for z,w in zip(cs,ws))
        return p*p*expit((Ee-s.mu_lepton)/s.baryons.T)*angular
    es=[c.ME,.53,.6,.8,1.,1.3,2.,3.,5.,10.,25.,60.]
    ps=[math.sqrt(E*E-c.ME*c.ME) for E in es]
    return a.K*sum(quad(f,lo,hi,epsabs=1e-30,epsrel=2e-10,limit=200)[0] for lo,hi in zip(ps[:-1],ps[1:]))
refs=[independent(k) for k in [64,128]]
values=[math.exp(a.log_coefficients(s,Enu,energy_order=k,angle_order=k)[1]) for k in [24,96]]
out=dict(Enu=Enu,independent_momentum_angular_orders=[64,128],independent_chi=refs,frozen_orders=[24,96],frozen_chi=values,
         relative_errors=[v/refs[-1]-1 for v in values],independent_refinement_error=abs(refs[0]/refs[1]-1),source_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
         disposition='Default24 exceeds original3e-6target here; endpoint extension remains unqualified')
Path(__file__).with_name('frozen_endpoint_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
