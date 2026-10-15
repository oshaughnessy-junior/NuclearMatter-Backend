"""Independent guard-size, held-out transfer and source normalization review."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad,solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantum_slab'))
import fixed_bath as b
import guarded_capture as g
from allowed_capture import K,ME,angular_factor,nodes
from free_density_response import log_response
from scipy.special import log_expit,logsumexp
checks=[];counterexamples=[];samples=[]
def check(name,actual,expected,tol=2e-8,scale=1e-300):
    error=float(abs(actual-expected)/max(abs(expected),scale))
    checks.append(dict(name=name,actual=float(actual),expected=float(expected),error=error,passed=error<tol))

matter=b.matter_state(2.,.002,.001)
opt=matter.optical('nue')
lj,lc,discard=g.log_coefficients(opt,60.,energy_order=72,angle_order=72,tail=50.)
checks.append(dict(name='original collapsed panel reproduced',passed=bool(discard),discarded=discard))
high=g.log_coefficients(opt,60.,energy_order=96,angle_order=96,tail=60.)
check('guarded rate refinement j',math.exp(lj),math.exp(high[0]),3e-5)
check('guarded rate refinement chi',math.exp(lc),math.exp(high[1]),3e-5)
cs,cws=nodes(72);sj=[];sc=[]
for panel in discard:
    c=panel['cosine'];cw=cws[int(np.argmin(abs(cs-c)))];Ee=.5*(panel['lo']+panel['hi']);width=panel['width']
    pe=math.sqrt((Ee-ME)*(Ee+ME));q=math.sqrt(60**2+pe**2-120*pe*c);omega=60-Ee
    common=math.log(K*cw*width*Ee*pe*angular_factor(Ee,c))
    sc.append(common+log_expit((Ee-opt.mu_lepton)/2)+log_response(opt.baryons,q,omega))
    sj.append(common+log_expit((opt.mu_lepton-Ee)/2)+log_response(opt.baryons.reverse(),q,-omega))
check('discarded panel midpoint relative measure j',math.exp(logsumexp(sj)-lj),0.,1e-12,1.)
check('discarded panel midpoint relative measure chi',math.exp(logsumexp(sc)-lc),0.,1e-12,1.)

for flavor,E in [('nue',20.),('antinue',.85)]:
    c=b.coefficients(matter,flavor,E,order=48)
    tau=37.5;mu=.37;left,right=b.boundary_occupations(E);length=tau/c['relaxation_cm_inv']
    flux=b.fluxes(c,length,left,right);eq=c['feq'];A=b.LIGHT*b.DOS*E*E*1e39
    # Integrate physical collision ODE in a scaled occupation variable.
    occscale=max(eq,left,1e-300)
    def rhs(t,y):
        f=y[0]*occscale
        return [(c['j_s']*(1-f)-c['chi_s']*f)/(c['rate_s']*mu*occscale)]
    end=solve_ivp(rhs,(0.,tau),[left/occscale],rtol=1e-11,atol=max(eq/occscale*1e-11,1e-24),method='DOP853').y[0,-1]*occscale
    check(flavor+' heldout physical ray ODE',b.ray_occupation(tau,mu,tau,eq,left,right),end,2e-8,1e-25)
    # Independent outgoing angular integral of the two boundary-value rays.
    def outgoing(inc):
        scale=max(eq,inc*math.exp(-tau),1e-300)
        value=quad(lambda u:u*(eq+(inc-eq)*math.exp(-tau/u))/scale,0,1,epsabs=1e-13,epsrel=1e-11)[0]*scale
        return A*.5*value
    check(flavor+' heldout left face normalization',flux['outgoing_left'],outgoing(right),2e-10)
    check(flavor+' heldout right face normalization',flux['outgoing_right'],outgoing(left),2e-10)
    # Direct angular/depth integral of C/rate. Use actual ray occupations,
    # not E2/E3 source identities or a sum of copied boundary fluxes.
    def collision_depth(t):
        def ray_C(u):
            fleft=eq+(left-eq)*math.exp(-t/u)
            fright=eq+(right-eq)*math.exp(-(tau-t)/u)
            return ((c['j_s']*(1-fleft)-c['chi_s']*fleft)+(c['j_s']*(1-fright)-c['chi_s']*fright))/c['rate_s']
        return quad(ray_C,0,1,epsabs=1e-13,epsrel=1e-9,limit=200)[0]/2
    edges=[0.,.01,.1,1.,5.,tau/2,tau-5,tau-1,tau-.1,tau-.01,tau]
    volume=A*sum(quad(collision_depth,lo,hi,epsabs=1e-12,epsrel=1e-8,limit=200)[0] for lo,hi in zip(edges[:-1],edges[1:]))
    check(flavor+' direct volume collision conservation',flux['radiation_number_gain'],volume,2e-8)
    check(flavor+' boundary number conservation',flux['outgoing_both']-flux['incoming_left']-flux['incoming_right'],volume,2e-8)
    check(flavor+' energy bath exchange',flux['bath_energy_source_MeV'],-E*volume,2e-8)
    samples.append(dict(flavor=flavor,E=E,tau=tau,feq=eq,flux=flux))

try:
    bad=b.coefficients(matter,'nue',60.,order=72,tail=-1.)
except ValueError:
    checks.append(dict(name='invalid tail rejected before fallback',passed=True))
else:
    counterexamples.append(dict(kind='invalid negative tail accepted through guard fallback',guard=bad['capture_panel_guard']))

out=dict(passed=all(x['passed'] for x in checks),checks=checks,counterexamples=counterexamples,samples=samples,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(b.__file__),Path(g.__file__)]})
Path(__file__).with_name('quantum_slab_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],counterexamples=counterexamples),indent=2))
