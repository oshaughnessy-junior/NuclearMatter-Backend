"""Run bounded analytic-limit, convergence, and thermodynamic checks."""
from __future__ import annotations
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from lepton_reference import state_mu, state_density, HBARC, ELECTRON_MASS, MUON_MASS, thermal_energy_density

results = []

def check(name, actual, expected, tolerance, scale=None):
    err = abs(actual-expected)/max(abs(expected), scale or 1e-300)
    results.append(dict(name=name, actual=float(actual), expected=float(expected),
                        scaled_error=float(err), tolerance=tolerance, passed=bool(err<tolerance)))

# Exact massless gas formulas, independent of quadrature.
for T, mu in [(1.,0.), (10.,3.), (0.01,100.), (10.,-3.)]:
    st=state_mu(T,mu,mass=0.)
    h3=HBARC**3
    pressure=(mu**4/(12*math.pi**2)+mu*mu*T*T/6+7*math.pi**2*T**4/180)/h3
    density=(mu**3/(3*math.pi**2)+mu*T*T/3)/h3
    entropy=(mu*mu*T/3+7*math.pi**2*T**3/45)/h3
    a=(mu*mu/math.pi**2+T*T/3)/h3
    b=2*mu*T/(3*h3)
    cv=T*((mu*mu/3+7*math.pi**2*T*T/15)/h3-b*b/a)
    for name,actual,expected in [('P',st.pressure,pressure),('n',st.net_density,density),
                                 ('s',st.entropy_density,entropy),('cv_n',st.cv_fixed_net_density,cv)]:
        check(f'massless T={T} mu={mu} {name}',actual,expected,2e-8,1e-30)

# Nonrelativistic dilute analytic limit at fixed mu, relative correction O(T/m).
m=MUON_MASS; T=1e-4; eta=-15.; mu=m+eta*T
st=state_mu(T,mu,m)
mb=2*(m*T/(2*math.pi))**1.5*math.exp(eta)/HBARC**3
check('nonrelativistic Maxwell-Boltzmann density',st.net_density,mb,3e-6)
check('nonrelativistic Maxwell-Boltzmann P/nT',st.pressure/(st.net_density*T),1.,2e-6)
check('nonrelativistic Maxwell-Boltzmann cv/n',st.cv_fixed_net_density/st.net_density,1.5,5e-6)

# Massive zero-temperature independent closed form (finite-T error O(T^2)).
m=MUON_MASS; mu=180.; T=1e-3; st=state_mu(T,mu,m)
p=math.sqrt(mu*mu-m*m); h3=HBARC**3
n0=p**3/(3*math.pi**2*h3)
e0=(p*mu*(2*p*p+m*m)-m**4*math.log((p+mu)/m))/(8*math.pi**2*h3)
p0=(p*mu*(2*p*p-3*m*m)+3*m**4*math.log((p+mu)/m))/(24*math.pi**2*h3)
for label,actual,expected in [('n',st.net_density,n0),('energy',st.energy_density,e0),('P',st.pressure,p0)]:
    check('massive cold '+label,actual,expected,5e-9)
check('massive degenerate thermal increment',thermal_energy_density(st),st.cv_fixed_net_density*T/2,1e-7)
check('massive degenerate Sommerfeld cv/nT',st.cv_fixed_net_density/(n0*T),math.pi**2*mu/(p*p),1e-7)

samples=[]
for label,T,n,m in [('pair_rich_e',10.,1e-6,ELECTRON_MASS),
                     ('degenerate_e',0.001,0.01,ELECTRON_MASS),
                     ('warm_mu',20.,0.005,MUON_MASS),
                     ('cold_mu',0.01,0.002,MUON_MASS),
                     ('dilute_e',0.1,1e-12,ELECTRON_MASS)]:
    st=state_density(T,n,m)
    samples.append(dict(label=label, thermal_energy_density=thermal_energy_density(st), **st.asdict()))
    check(label+' density inversion',st.net_density,n,1e-9)
    check(label+' Euler identity',st.energy_density+st.pressure,
          T*st.entropy_density+st.chemical_potential*n,2e-9)
    refined=state_mu(T,st.chemical_potential,m,rtol=5e-13,tail=80.)
    for field in ['pressure','energy_density','entropy_density','cv_fixed_net_density']:
        check(label+' quadrature/tail '+field,getattr(st,field),getattr(refined,field),5e-8)
    # Fixed-density Cv finite difference checks derivative of *total* energy.
    # Cold FD of huge energy is deliberately excluded from the tight gate.
    h=(5e-2 if abs(st.chemical_potential)/T > 1e3 else 2e-3)*T
    minus=state_density(T-h,n,m); plus=state_density(T+h,n,m)
    cv_fd=(plus.energy_density-minus.energy_density)/(2*h)
    check(label+' cv fixed-n finite difference',st.cv_fixed_net_density,cv_fd,5e-4)
    dmu_fd=(plus.chemical_potential-minus.chemical_potential)/(2*h)
    check(label+' dmu/dT fixed-n FD',st.dmu_dtemperature_fixed_net_density,dmu_fd,5e-4)
    # Maxwell relation: ds/dmu = dn/dT at fixed T.
    if label != 'degenerate_e':
        hm=1e-4*max(T,abs(st.chemical_potential),1e-3)
        sm=state_mu(T,st.chemical_potential-hm,m); sp=state_mu(T,st.chemical_potential+hm,m)
        ds=(sp.entropy_density-sm.entropy_density)/(2*hm)
        check(label+' Maxwell ds/dmu=dn/dT',ds,st.dn_dtemperature_fixed_mu,2e-5)

# Independent review found fixed-MeV root tolerance losing tiny net charges.
for T in (0.01, 1., 10.):
    for n in (1e-18, 1e-24, 1e-30):
        for sign in (-1., 1.):
            st = state_density(T, sign*n, 0.)
            check(f'pair symmetric inversion T{T} n{sign*n}', st.net_density, sign*n, 1e-9)

report={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'scipy':scipy.__version__,'checks':results,'sample_states':samples,
        'passed':all(x['passed'] for x in results)}
path=Path(__file__).with_name('validation_results.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(x['passed'] for x in results)}/{len(results)} checks pass")
for x in results:
    if not x['passed']: print('FAIL',x)
print(path)
raise SystemExit(0 if report['passed'] else 1)
