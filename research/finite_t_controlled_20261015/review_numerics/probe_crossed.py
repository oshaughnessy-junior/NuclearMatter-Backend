"""Independent threshold/adaptive checks of crossed decay optical channel."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'crossed_opacity'))
import crossed_decay as d
import allowed_capture as a
from free_density_response import response
import static_capture as c
checks=[];samples=[]
def check(name,actual,expected,tol=3e-6):
    error=float(np.max(abs(np.asarray(actual)/expected-1)))
    checks.append(dict(name=name,actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist(),relative_error=error,passed=error<tol))

def independent_crossed_j(s,Enu,nangle=96):
    # Adaptive scalar electron-momentum integral and separate high-order
    # lepton-angle rule; no author endpoint/panel generator is reused.
    cs,cws=leggauss(nangle)
    def f(p):
        Ee=math.hypot(p,c.ME)
        angular=sum(w*a.angular_factor(Ee,z)*response(s.baryons,math.sqrt(Enu*Enu+p*p+2*Enu*p*z),-Enu-Ee) for z,w in zip(cs,cws))
        # dEe*Ee*pe = p² dp, giving a smooth true momentum endpoint.
        return p*p*expit((Ee-s.mu_e)/s.baryons.T)*angular
    energies=[c.ME,.53,.6,.8,1.,1.3,2.,3.,5.,10.,25.,60.]
    ps=[math.sqrt(E*E-c.ME*c.ME) for E in energies]
    return a.K*sum(quad(f,lo,hi,epsabs=1e-24,epsrel=1e-9,limit=150)[0] for lo,hi in zip(ps[:-1],ps[1:]))

for s,Enu in [(d.DecayState(d.PairState(3,939,c.Q,0,c.Q+20,5),5),1.),
              (d.DecayState(d.PairState(2,939,c.Q,0,c.Q+8,1),1.3),.8)]:
    vals=[d.log_coefficients(s,Enu,energy_order=k,angle_order=k) for k in [24,48]]
    j=math.exp(vals[-1][0]);other=independent_crossed_j(s,Enu)
    check(f'crossed momentum-adaptive Enu{Enu}',j,other)
    check(f'crossed order refinement Enu{Enu}',np.exp(vals[0]),np.exp(vals[-1]))
    lhs=vals[-1][0]-vals[-1][1]+(Enu-s.mu_antineutrino)/s.baryons.T
    checks.append(dict(name=f'crossed KMS Enu{Enu}',absolute_error=abs(lhs),passed=abs(lhs)<2e-11))
    ext=d.log_coefficients(s,Enu,energy_order=48,angle_order=48,tail=70)
    check(f'crossed tail refinement Enu{Enu}',np.exp(vals[-1]),np.exp(ext))
    checks.append(dict(name=f'finite thermal tail Enu{Enu}',log_j=vals[-1][0],passed=math.isfinite(vals[-1][0])))
    samples.append(dict(T=s.baryons.T,Enu=Enu,log_coefficients=vals[-1],independent_j=other))

def heavy_state(T,mu,nn,np_,M):
    thermal=2*(M*T/(2*math.pi))**1.5/c.HBARC**3
    return d.DecayState(d.PairState(T,M,c.Q,0,c.Q+T*math.log(nn/thermal),T*math.log(np_/thermal)),mu)

# Held-out heavy common-mass integrated normalization, using both number and
# neutrino-energy moments. Electron endpoint map is always the author's fixed one.
nn,np_=8e-10,2e-10
s=heavy_state(1.5,1.1,nn,np_,1e7)
threshold=c.Q-c.ME;width=c.Q*math.sqrt(s.baryons.T/s.baryons.mass)
upper=threshold+.05
edges=sorted({0.,.2,.5,threshold,upper}|{threshold+k*width for k in [-20,-8,-2,2,8,20] if 0<threshold+k*width<upper})
def moments(order):
    xs,ws=leggauss(order);out=np.zeros(2)
    for lo,hi in zip(edges[:-1],edges[1:]):
        E=.5*(lo+hi)+.5*(hi-lo)*xs
        for enu,w in zip(E,.5*(hi-lo)*ws):
            j=math.exp(d.log_coefficients(s,float(enu),energy_order=24,angle_order=16)[0])
            out+=c.DOS_PREFACTOR*w*j*np.array([enu**2,enu**3])
    return out
coarse=moments(12);fine=moments(24)
expected=nn*c.target_moments(s.baryons.T,s.mu_e,'neutron_decay',192)
check('held-out heavyM number+energy normalization',fine,expected,1e-5)
check('held-out heavyM outer moment refinement',coarse,fine,2e-6)
samples.append(dict(heavy_mass=1e7,T=1.5,mu_e=1.1,moments=fine.tolist(),static_moments=expected.tolist()))

out=dict(passed=all(x['passed'] for x in checks),checks=checks,samples=samples,
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(d.__file__),Path(a.__file__),ROOT/'response_reference/free_density_response.py']})
Path(__file__).with_name('crossed_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']]),indent=2))
