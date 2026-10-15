"""Targeted independent response and prescribed-density physics checks."""
import sys,json,math
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'response_reference'),str(ROOT/'prescribed_density')]
import free_density_response as r
import element as el

response_checks=[]
density_checks=[]
def add(bucket,name,error,tol):
    bucket.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))

# 100-digit elementary reference targets numerical branch boundaries directly.
worst=0.
for a in [-45.,-40.000001,-40.,-39.999999,-5.,0.,5.,40.,100.]:
    for z in [-100.,-1.,-1e-5,-.999e-5,0.,.999e-5,1e-5,1.,100.]:
        with localcontext() as ctx:
            ctx.prec=100
            aa,zz=Decimal(str(a)),Decimal(str(z))
            if z==0:
                expected=-(1+(-aa).exp()).ln()
            else:
                numerator=(1+aa.exp()).ln()-(1+(aa-zz).exp()).ln()
                expected=(numerator/(1-(-zz).exp())).ln()
            error=abs(r._log_integral(a,z)-float(expected))
            worst=max(worst,error)
add(response_checks,'81 high-precision branch-boundary log-integral comparisons',worst,2e-10)

# Integrated charged-response commutator: cross Pauli terms cancel, leaving na-nb.
state=r.PairState(3.,939.,939.,942.,969.,947.)
q=70.
def density(nu):
    return quad(lambda p:p*p*expit((nu-p*p/(2*state.mass))/state.T),
                0,1200,epsabs=1e-8,epsrel=2e-12)[0]/(math.pi**2*r.HBARC**3)
expected=density(state.kinetic_mu_a)-density(state.kinetic_mu_b)
measured=quad(lambda w:(r.response(state,q,w)-r.response(state.reverse(),q,w))/(2*math.pi),
              -200,200,epsabs=1e-12,epsrel=2e-11)[0]
add(response_checks,'charged equal-time commutator na-minus-nb',abs(measured/expected-1),1e-9)

# Differentiate complete EOS in (T,mu,n), solve charge+energy system independently.
for factor in [.5,2.]:
    for weak_on in [False,True]:
        tr=el.Trajectory(factor,weak_on=weak_on)
        t=.035;n,ndot=tr.density(t);T=1.8;Y=.4
        mu=el.lepton.state_density(T,n*Y).chemical_potential
        rhs=tr.rhs(t,[T,mu,0,0,0,0])
        def fields(tt,mm,nn):
            s=el.eos(tt,nn,.4,mm)
            return np.array([s['u'],s['Y']])
        hT,hmu,hn=T*1e-5,max(abs(mu),1)*1e-5,n*1e-5
        J=np.column_stack([(fields(T+hT,mu,n)-fields(T-hT,mu,n))/(2*hT),
                           (fields(T,mu+hmu,n)-fields(T,mu-hmu,n))/(2*hmu)])
        dn=(fields(T,mu,n+hn)-fields(T,mu,n-hn))/(2*hn)
        target=np.array([rhs[5]-rhs[4],rhs[3]-rhs[2]])-dn*ndot
        solved=np.linalg.solve(J,target)
        add(density_checks,f'full finite-difference source solve factor={factor} weak={weak_on}',
            max(abs(solved-rhs[:2])/np.maximum(1.,abs(rhs[:2]))),2e-7)

for name,checks in [('response_review_results.json',response_checks),
                    ('prescribed_density_review_results.json',density_checks)]:
    payload=dict(scope='Independent targeted probes, no ODE replay',checks=checks,
                 all_passed=all(x['passed'] for x in checks))
    Path(__file__).with_name(name).write_text(json.dumps(payload,indent=2)+'\n')
    print(name,json.dumps(payload))
