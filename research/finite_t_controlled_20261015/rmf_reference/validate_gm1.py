"""Independent-convention cold source checks and finite-T thermodynamic gates."""
from __future__ import annotations
import ast
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.optimize import brentq
from gm1_finite_t import *

ROOT=Path(__file__).resolve().parent
checks=[]; start=time.monotonic()
def check(label,actual,expected,tol,scale=1e-20):
    error=abs(actual-expected)/max(abs(expected),scale)
    checks.append(dict(name=label,actual=float(actual),expected=float(expected),scaled_error=float(error),tolerance=tol,passed=bool(error<tol)))

# Execute only the actual pinned pure cold formulas, with NumPy as array API.
# No production package installation/monkeypatch is needed. No retyped formulas.
source=ast.parse((ROOT/'source_snapshot/rmft.py').read_text())
source_constants={}
exec(compile(ast.parse((ROOT/'source_snapshot/constants.py').read_text()),'pinned_NMB_constants','exec'),source_constants)
check('pinned source HBARC',HBARC,source_constants['HBARC'],1e-15)
source_params={}
exec(compile(ast.Module(body=[x for x in source.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GM1' for t in x.targets)],type_ignores=[]),'pinned_NMB_GM1','exec'),source_params)
for key,value in GM1.items():check('pinned source parameter '+key,value,source_params['GM1'][key],1e-15)
names={'_kf','_rho_s_one','_eps_kin_one','_vector_fields','_eps_p_from_state','symmetry_energy'}
namespace={'jnp':np,'math':math,'_PI2':math.pi**2,'HBARC':HBARC}
exec(compile(ast.Module(body=[x for x in source.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]),'pinned_NMB_cold_formulas','exec'),namespace)
for n,Y in [(.08,.5),(.153,.5),(.32,.1),(.8,.05)]:
    s=cold_state(n,Y); phi=s.phi_mev/HBARC
    eps,pres,mun=namespace['_eps_p_from_state'](n,Y,phi,GM1,leptons=False)
    for name,a,b in [('energy',s.energy_density,eps),('pressure',s.pressure,pres),('mu_n',s.mu_n_mev,mun)]:
        check(f'pinned source cold n={n} Y={Y} {name}',a,b,3e-11)
    rs=namespace['_rho_s_one'](kf((1-Y)*n),s.dirac_mass_mev/HBARC)+namespace['_rho_s_one'](kf(Y*n),s.dirac_mass_mev/HBARC)
    check(f'pinned source scalar n={n} Y={Y}',rs,s.scalar_density,3e-12)

n0=brentq(lambda n:cold_state(n,.5).pressure,.12,.18,xtol=2e-13)
s0=cold_state(n0,.5); h=1e-5
K=9*(cold_state(n0+h,.5).pressure-cold_state(n0-h,.5).pressure)/(2*h)
S=float(namespace['symmetry_energy'](n0,s0.phi_mev/HBARC,GM1))
calibration=dict(n0=n0,binding=s0.energy_density/n0-939.,K=K,dirac_mass_fraction=s0.dirac_mass_mev/939.,symmetry_energy=S)
# Loose gates reflect rounded couplings/source calibration claims, not measured accuracy.
for key,expected,tol in [('n0',.153,.003),('binding',-16.3,.003),('K',300.,.01),('dirac_mass_fraction',.70,.003),('symmetry_energy',32.5,.005)]:
    check('GM1 calibration '+key,calibration[key],expected,tol)

# Independent momentum-variable fixed Gauss-Legendre quadrature, not the
# adaptive kinetic-energy routine used by the solver.
nodes,weights=np.polynomial.legendre.leggauss(96)
def momentum_reference(T,nu,mass):
    t=T/HBARC
    emax=max(mass,abs(nu))+60*t
    epoints=sorted(set([mass,emax]+[x for x in (abs(nu)-50*t,abs(nu),abs(nu)+50*t) if mass<x<emax]))
    moment=np.zeros(5)
    for elo,ehi in zip(epoints[:-1],epoints[1:]):
        lo=math.sqrt(max(0.,elo*elo-mass*mass));hi=math.sqrt(ehi*ehi-mass*mass)
        p=(hi+lo)/2+(hi-lo)*nodes/2
        E=np.sqrt(p*p+mass*mass);x=(E-nu)/t;y=(E+nu)/t
        a=expit(-x);b=expit(-y)
        entropy=lambda z:np.logaddexp(0.,-abs(z))+abs(z)*expit(-abs(z))
        values=np.array([p*p*(a-b),p*p*mass/E*(a+b),p*p*E*(a+b),p**4/(3*E)*(a+b),p*p*(entropy(x)+entropy(y))])
        moment+=(hi-lo)/2*(values@weights)/math.pi**2
    return dict(zip(('n','scalar','energy','pressure','entropy'),moment))

samples=[]
for n,T,Y in [(.16,1.,.5),(.16,10.,.1),(.32,30.,.1),(.8,20.,.05),(.08,5.,.5)]:
    label=f'n={n},T={T},Yp={Y}'
    s=state(n,T,Y); samples.append(s.asdict())
    check(label+' solved equations',s.max_normalized_equation_residual,0.,2e-9,1.)
    for tag,nu in [('neutron',s.kinetic_mu_n_mev/HBARC),('proton',s.kinetic_mu_p_mev/HBARC)]:
        ref=momentum_reference(T,nu,s.dirac_mass_mev/HBARC)
        adaptive=gas(T/HBARC,nu,s.dirac_mass_mev/HBARC)
        for field in ref:check(label+' independent p-Gauss '+tag+' '+field,adaptive[field],ref[field],2e-7)
    check(label+' Euler',s.energy_density+s.pressure,T*s.entropy_density+n*((1-Y)*s.mu_n_mev+Y*s.mu_p_mev),2e-10)
    refined=state(n,T,Y,rtol=2e-12,tail=80.,initial=s)
    for field in ['phi_mev','energy_density','pressure','entropy_density','mu_n_mev','mu_p_mev']:
        check(label+' quadrature '+field,getattr(s,field),getattr(refined,field),3e-8)
    hn=2e-4*n; ht=2e-3*T; hy=1e-4
    nm=state(n-hn,T,Y,initial=s); np_=state(n+hn,T,Y,initial=s)
    tm=state(n,T-ht,Y,initial=s); tp=state(n,T+ht,Y,initial=s)
    ym=state(n,T,Y-hy,initial=s); yp=state(n,T,Y+hy,initial=s)
    dfdt=(tp.free_energy_density-tm.free_energy_density)/(2*ht)
    dfdn=(np_.free_energy_density-nm.free_energy_density)/(2*hn)
    dfdy=(yp.free_energy_density-ym.free_energy_density)/(2*hy)
    check(label+' entropy=-df/dT',-dfdt,s.entropy_density,3e-5)
    check(label+' df/dnB fixed Y',dfdn,(1-Y)*s.mu_n_mev+Y*s.mu_p_mev,3e-6)
    check(label+' df/dYp',dfdy/n,s.mu_p_mev-s.mu_n_mev,3e-5,1.)
    check(label+' pressure from f',n*dfdn-s.free_energy_density,s.pressure,3e-5,1.)
    ds_dn=(np_.entropy_density-nm.entropy_density)/(2*hn)
    mubar=lambda x:(1-Y)*x.mu_n_mev+Y*x.mu_p_mev
    dmubar_dt=(mubar(tp)-mubar(tm))/(2*ht)
    check(label+' Maxwell ds/dnB=-dmu_bar/dT',ds_dn,-dmubar_dt,4e-5,1e-6)
    cv=(tp.energy_density-tm.energy_density)/(2*ht)
    check(label+' positive frozen Cv',float(cv>0),1.,1e-12)
    check(label+' kinetic/scalar/vector sum',s.kinetic_energy_density+s.scalar_field_energy_density+s.vector_field_energy_density,s.energy_density,1e-12)
    # Independent route: keep n_i fixed while perturbing scalar field and re-solving nu_i.
    phi=s.phi_mev/HBARC; M=939./HBARC; hp=2e-5*M
    def free_phi(phi_):
        gn=fixed_density_gas(T,(1-Y)*n,M-phi_)
        gp=fixed_density_gas(T,Y*n,M-phi_)
        return (gn['energy']+gp['energy']+scalar_potential(phi_))*HBARC-T*(gn['entropy']+gp['entropy'])
    f0=free_phi(phi); fm=free_phi(phi-hp); fp=free_phi(phi+hp)
    gradient=(fp-fm)/(2*hp)
    curvature=(fp-2*f0+fm)/(hp*hp)
    check(label+' free-energy scalar stationarity',gradient,0.,3e-7,HBARC*n)
    check(label+' local scalar curvature positive',float(curvature>0),1.,1e-12)

# Cross-species Maxwell symmetry is independent of the frozen-Y density derivative.
for n,T,Y in [(.16,10.,.1),(.32,30.,.1)]:
    nn=n*(1-Y); np_=n*Y; hi=1e-4*min(nn,np_)
    def at_species(nn_,np__):
        return state(nn_+np__,T,np__/(nn_+np__))
    nplus=at_species(nn+hi,np_); nminus=at_species(nn-hi,np_)
    pplus=at_species(nn,np_+hi); pminus=at_species(nn,np_-hi)
    dmun_dnp=(pplus.mu_n_mev-pminus.mu_n_mev)/(2*hi)
    dmup_dnn=(nplus.mu_p_mev-nminus.mu_p_mev)/(2*hi)
    check(f'cross species Maxwell n={n} T={T}',dmun_dnp,dmup_dnn,2e-5)

# Primal cold gate before interpreting finite-T derivatives.
for n,Y in [(.16,.5),(.32,.1),(.8,.05)]:
    cold=cold_state(n,Y); warm=state(n,.01,Y)
    for field in ['phi_mev','energy_density','pressure','mu_n_mev','mu_p_mev']:
        check(f'T->0 n={n} Y={Y} '+field,getattr(warm,field),getattr(cold,field),2e-6,1.)
    check(f'cold entropy vanishes n={n}',float(warm.entropy_density>0 and warm.entropy_density/n<.01),1.,1e-12)

# Explicit free-energy-based low-T approximation with derivative-consistent pressure.
asymptotic_samples=[]
for n,Y in [(.16,.5),(.32,.1),(.8,.05)]:
    T=.1;c=cold_state(n,Y);s=state(n,T,Y);fit=sommerfeld_thermal(n,T,Y)
    asymptotic_samples.append(dict(nB=n,T=T,Yp=Y,**fit))
    for name,a,b in [('entropy',s.entropy_density,fit['entropy_density']),
                     ('thermal energy',s.energy_density-c.energy_density,fit['delta_energy_density']),
                     ('thermal free energy',s.free_energy_density-c.free_energy_density,fit['delta_free_energy_density']),
                     ('thermal pressure',s.pressure-c.pressure,fit['delta_pressure'])]:
        check(f'qualified Sommerfeld n={n} Y={Y} '+name,a,b,2e-4)

# A finite-T continuation diagnostic, independent initial seeds on same branch.
anchor=state(.32,10.,.1)
continued=state(.32,30.,.1,initial=anchor)
direct=state(.32,30.,.1)
check('warm continuation scalar branch',continued.phi_mev,direct.phi_mev,1e-10)

# Explicit counterexample to equating a local scalar root with phase stability.
spin_n=.06; spin_T=1.; spin_Y=.5; hn=1e-5
spin_dPdn=(state(spin_n+hn,spin_T,spin_Y).pressure-state(spin_n-hn,spin_T,spin_Y).pressure)/(2*hn)
check('subnuclear example flags mechanical instability',float(spin_dPdn<0),1.,1e-12)

report=dict(passed=all(c['passed'] for c in checks),checks=checks,saturation=calibration,sample_states=samples,sommerfeld_samples=asymptotic_samples,
            spinodal_example=dict(nB=spin_n,temperature=spin_T,Yp=spin_Y,dP_dn_fixed_T_Yp=spin_dPdn),
            runtime_seconds=time.monotonic()-start,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__)
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
print(json.dumps(calibration,indent=2))
for c in checks:
    if not c['passed']:print('FAIL',c)
raise SystemExit(0 if report['passed'] else 1)
