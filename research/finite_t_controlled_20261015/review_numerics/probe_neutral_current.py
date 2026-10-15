"""Independent q/omega NC moments, event-pair grid and input audit."""
from pathlib import Path
import sys,json,math,hashlib,warnings
import numpy as np
from scipy.integrate import quad,quad_vec
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neutral_current'))
import free_nc as n
checks=[];counterexamples=[];samples=[]
def check(name,actual,expected,tol=2e-5,scale=1e-25):
    error=float(np.linalg.norm(np.asarray(actual)-expected)/max(np.linalg.norm(np.asarray(expected)),scale))
    checks.append(dict(name=name,actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist(),error=error,passed=error<tol))

def independent_moments(species,Ein,couplings):
    pair=n.pair_from_species(species)
    qmax=2*Ein+70*species.T
    def outer(q):
        lower=-q;upper=min(q,2*Ein-q)
        center=q*q/(2*species.mass)
        spread=q*math.sqrt(species.T/species.mass+2*max(species.kinetic_mu,0)/species.mass)
        points=sorted({lower,upper}|{x for x in [0.,center,center-8*spread,center+8*spread] if lower<x<upper})
        def inner(omega):
            Eout=Ein-omega
            c=(Ein*Ein+Eout*Eout-q*q)/(2*Ein*Eout)
            c=max(-1.,min(1.,c))
            value=n.kernel(pair,Ein,Eout,c,couplings)*q/(Ein*Eout)
            return value*np.array([1.,1-c,omega])
        return sum((quad_vec(inner,lo,hi,epsabs=1e-15,epsrel=2e-10,limit=250)[0] for lo,hi in zip(points[:-1],points[1:])),np.zeros(3))
    qwidth=2*Ein*math.sqrt(species.T/species.mass+2*max(species.kinetic_mu,0)/species.mass)
    edges=sorted({0.,Ein,2*Ein,min(4*Ein,qmax),qmax}|{2*Ein+k*qwidth for k in [-12,-4,-1,1,4,12] if 0<2*Ein+k*qwidth<qmax})
    return sum((quad_vec(outer,lo,hi,epsabs=1e-12,epsrel=2e-8,limit=200)[0] for lo,hi in zip(edges[:-1],edges[1:])),np.zeros(3))

s=n.state_density(3.,3e-4)
mom=independent_moments(s,7.,n.NEUTRON)
author=n.moments(s,7.,n.NEUTRON,angle_order=48,energy_rtol=2e-11,tail=60.)
for i,key in enumerate(['rate_s','angular_transport_rate_s','energy_to_bath_MeV_s']):check('independent q/omega '+key,author[key],mom[i],3e-5)
samples.append(dict(species=s.asdict(),Ein=7.,independent_moments=mom.tolist(),reference=author))

# A held-out higher-M energy and thermal state checks absolute elastic units.
heavy=n.state_density(1.7,3e-6,mass=1e7)
hm=independent_moments(heavy,13.,n.NEUTRON)
cross=n.bare_static_cross_sections(13.,n.NEUTRON)
conversion=heavy.density*1e39*n.LIGHT
check('heldout heavyM elastic total absolute',hm[0]/conversion,cross['total_cm2'],2e-5,1e-50)
check('heldout heavyM elastic transport absolute',hm[1]/conversion,cross['transport_cm2'],2e-5,1e-50)
samples.append(dict(heavy_mass=1e7,T=1.7,Ein=13.,independent_moments=hm.tolist(),static_cross_sections=cross))

pair=n.pair_from_species(s)
for Ei,Ef,c in [(7.,5.3,.23),(7.,9.1,-.41),(3.,3.0000001,.7)]:
    check('DOS ratio KMS '+str((Ei,Ef,c)),n.log_kernel(pair,Ef,Ei,c)-n.log_kernel(pair,Ei,Ef,c),2*math.log(Ei/Ef)-(Ei-Ef)/s.T,2e-11,1.)

E=np.array([1.3,3.7,8.1,13.4]);w=np.array([.7,2.2,4.9,6.1]);density_weights=n.DOS*E**2*w
for equilibrium in [True,False]:
    f=expit((2.5-E)/(s.T if equilibrium else 1.6*s.T))
    author=n.discrete_collision(s,E,w,f,angle_order=64)
    direct=np.zeros(len(E));bath=0.;gross=0.
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            # Direct unordered-event construction with adaptive angular integrals.
            forward=density_weights[i]*w[j]*quad(lambda c:n.kernel(pair,E[i],E[j],c),-1,1,epsabs=1e-14,epsrel=1e-11)[0]*f[i]*(1-f[j])
            reverse=density_weights[j]*w[i]*quad(lambda c:n.kernel(pair,E[j],E[i],c),-1,1,epsabs=1e-14,epsrel=1e-11)[0]*f[j]*(1-f[i])
            net=forward-reverse;direct[i]-=net;direct[j]+=net
            bath+=(E[i]-E[j])*net;gross+=forward+reverse
    tag='LTE' if equilibrium else 'nonequilibrium'
    check(tag+' independent event-pair sources',author['number_source'],direct,2e-10,max(gross,1e-30))
    check(tag+' independent event number conservation',sum(direct),0.,2e-13,max(gross,1e-30))
    check(tag+' independent event bath energy',author['net_neutrino_energy_source'],-bath,2e-10,max(gross*max(E),1e-30))
    if equilibrium:check('independent FD stationarity',np.max(abs(direct)),0.,2e-12,max(gross,1e-30))

for name,call in [
    ('moments negative tail',lambda:n.moments(s,7.,angle_order=8,tail=-1.)),
    ('moments negative radiation T',lambda:n.moments(s,7.,angle_order=8,radiation=(-1.,0.))),
    ('grid NaN occupation',lambda:n.discrete_collision(s,E,w,np.array([.2,np.nan,.3,.4]),angle_order=8)),
    ('grid NaN weight',lambda:n.discrete_collision(s,E,np.array([.7,np.nan,4.9,6.1]),np.full(4,.2),angle_order=8))]:
    try:
        with warnings.catch_warnings():warnings.simplefilter('ignore');value=call()
    except (ValueError,ArithmeticError):checks.append(dict(name=name+' rejected',passed=True))
    else:counterexamples.append(dict(kind='invalid input accepted',call=name,result={k:np.asarray(v).tolist() for k,v in value.items()}))

out=dict(valid_selected_cases_passed=all(c['passed'] for c in checks),checks=checks,counterexamples=counterexamples,samples=samples,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(n.__file__),ROOT/'response_reference/free_density_response.py',ROOT/'nr_baryon_eos/free_nucleons.py']})
Path(__file__).with_name('neutral_current_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(valid_cases_passed=out['valid_selected_cases_passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],invalid_calls_accepted=[c['call']for c in counterexamples]),indent=2))
