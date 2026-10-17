"""Uniform finite-T GM1 nucleonic RMF reference; deterministic NumPy/SciPy.

Fixed (nB [fm^-3], T [MeV], Yp); no beta equilibrium, leptons or crust.
NMB pinned conventions are retained: common bare M=939 MeV, no-sea Hartree,
sigma/omega/rho, no omega-rho cross coupling. Internal momenta/energies fm^-1.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq, least_squares
from scipy.special import expit

HBARC = 197.3269804
GM1 = dict(M_mev=939.0,C_sigma=11.785,C_omega=7.148,C_rho=4.410,b=.002947,c=-.001070)
PI2 = math.pi**2


def kf(n):
    return (3*PI2*n)**(1/3)


def scalar_potential(phi, params=GM1):
    M=params['M_mev']/HBARC
    return phi*phi/(2*params['C_sigma'])+params['b']*M*phi**3/3+params['c']*phi**4/4


def scalar_gradient(phi, params=GM1):
    M=params['M_mev']/HBARC
    return phi/params['C_sigma']+params['b']*M*phi*phi+params['c']*phi**3


def cold_gas(n, mass):
    """Independent closed-form zero-T moments, fm units; spin degeneracy 2."""
    p=kf(n); e=math.hypot(p,mass); log=math.asinh(p/mass)
    scalar=mass*(p*e-mass*mass*log)/(2*PI2)
    energy=(p*e*(2*p*p+mass*mass)-mass**4*log)/(8*PI2)
    pressure=(p*e*(2*p*p-3*mass*mass)+3*mass**4*log)/(24*PI2)
    return dict(n=n,scalar=scalar,energy=energy,pressure=pressure,entropy=0.,antiparticles=0.,nu=e)


def gas(T, nu, mass, *, rtol=2e-11, tail=60.):
    """Pair-equilibrated relativistic ideal nucleons at T,nu,m in fm^-1.

    Scalar density includes thermal antiparticles; vacuum Dirac sea excluded.
    nu is the kinetic chemical potential, before vector-field shift.
    """
    if T <= 0 or mass <= 0:
        raise ValueError('Require T,m > 0')
    end=max(0.,abs(nu)-mass)+tail*T
    panels=sorted(set([0.,end]+[x for x in (abs(nu)-mass-tail*T,abs(nu)-mass,abs(nu)-mass+tail*T) if 0<x<end]))
    def integrand(k):
        E=mass+k; p=math.sqrt(k*(k+2*mass)); jac=p*E/PI2
        x=(E-nu)/T; y=(E+nu)/T
        a=expit(-x); b=expit(-y)
        def entropy(z):
            z=abs(z)
            return np.logaddexp(0.,-z)+z*expit(-z)
        return jac*np.array([a-b,mass/E*(a+b),E*(a+b),p*p/(3*E)*(a+b),entropy(x)+entropy(y),b])
    out=np.zeros(6)
    for lo,hi in zip(panels[:-1],panels[1:]):
        val,err,info=quad_vec(integrand,lo,hi,epsabs=1e-15,epsrel=rtol,limit=300,full_output=True)
        if not info.success and not (info.status==2 and err <= max(1e-15,rtol*np.max(np.abs(val)))):
            raise ArithmeticError(f'Quadrature failure: {info.message}')
        out+=val
    return dict(zip(('n','scalar','energy','pressure','entropy','antiparticles'),map(float,out)),nu=nu)


@dataclass(frozen=True)
class NuclearState:
    nB: float
    temperature: float
    Yp: float
    phi_mev: float
    omega_mev: float
    rho_mev: float
    dirac_mass_mev: float
    density_defined_landau_n_mev: float
    density_defined_landau_p_mev: float
    kinetic_mu_n_mev: float
    kinetic_mu_p_mev: float
    mu_n_mev: float
    mu_p_mev: float
    energy_density: float
    pressure: float
    free_energy_density: float
    entropy_density: float
    kinetic_energy_density: float
    scalar_field_energy_density: float
    vector_field_energy_density: float
    scalar_density: float
    antiparticle_density: float
    max_normalized_equation_residual: float
    branch: str='local GM1 branch continued from cold first scalar upcrossing'
    phase_assessment: str='uniform local solution only; mechanical stability/coexistence unassessed'

    def asdict(self): return asdict(self)


def _assemble(nB, T_mev, Yp, phi, n, p, params, residual):
    M=params['M_mev']/HBARC; mass=M-phi
    W=params['C_omega']*nB; R=.5*params['C_rho']*(2*Yp-1)*nB
    us=scalar_potential(phi,params)
    uv=W*W/(2*params['C_omega'])+R*R/(2*params['C_rho'])
    kin=n['energy']+p['energy']; entropy=n['entropy']+p['entropy']
    energy=(kin+us+uv)*HBARC
    pressure=(n['pressure']+p['pressure']-us+uv)*HBARC
    return NuclearState(nB,T_mev,Yp,phi*HBARC,W*HBARC,R*HBARC,mass*HBARC,
        math.hypot(kf((1-Yp)*nB),mass)*HBARC,math.hypot(kf(Yp*nB),mass)*HBARC,
        n['nu']*HBARC,p['nu']*HBARC,(n['nu']+W-.5*R)*HBARC,(p['nu']+W+.5*R)*HBARC,
        energy,pressure,energy-T_mev*entropy,entropy,kin*HBARC,us*HBARC,uv*HBARC,
        n['scalar']+p['scalar'],n['antiparticles']+p['antiparticles'],residual)


def cold_state(nB, Yp, params=GM1):
    """Find first negative-to-positive cold scalar residual root in0<Phi<.985M."""
    _validate(nB,0.,Yp,params)
    M=params['M_mev']/HBARC; nn=(1-Yp)*nB; np_=Yp*nB
    def fun(phi):
        return scalar_gradient(phi,params)-cold_gas(nn,M-phi)['scalar']-cold_gas(np_,M-phi)['scalar']
    grid=np.linspace(0.,.985*M,160)
    for lo,hi in zip(grid[:-1],grid[1:]):
        if fun(lo)<0<=fun(hi):
            phi=brentq(fun,lo,hi,xtol=2e-14,rtol=2e-14)
            return _assemble(nB,0.,Yp,phi,cold_gas(nn,M-phi),cold_gas(np_,M-phi),params,abs(fun(phi))/nB)
    raise ArithmeticError('No physical first scalar upcrossing in declared mass branch')


def _validate(nB,T,Yp,params):
    if not all(math.isfinite(x) for x in (nB,T,Yp)) or nB<=0 or T<0 or not 0<Yp<1:
        raise ValueError('Require finite nB>0, T>=0, 0<Yp<1; endpoints not qualified')
    if params.get('Lambda_v',0.) != 0.:
        raise ValueError('Omega-rho cross coupling unsupported')


def state(nB,T,Yp,*,params=GM1,rtol=2e-11,tail=60.,initial=None):
    """Finite-T solve at frozen independent composition; energies MeV/fm^3.

    Warm solves initialize at the qualified cold root or a supplied nearby state.
    Continuation checks and response derivatives remain part of validation.
    """
    _validate(nB,T,Yp,params)
    if T==0: return cold_state(nB,Yp,params)
    M=params['M_mev']/HBARC; tfm=T/HBARC
    anchor=cold_state(nB,Yp,params) if initial is None else initial
    guess=np.array([anchor.phi_mev,anchor.kinetic_mu_n_mev,anchor.kinetic_mu_p_mev])/HBARC
    nn=(1-Yp)*nB; np_=Yp*nB
    def equations(x):
        phi,nun,nup=x; mass=M-phi
        gn=gas(tfm,nun,mass,rtol=rtol,tail=tail)
        gp=gas(tfm,nup,mass,rtol=rtol,tail=tail)
        return np.array([(scalar_gradient(phi,params)-gn['scalar']-gp['scalar'])/nB,
                         (gn['n']-nn)/nn,(gp['n']-np_)/np_])
    sol=least_squares(equations,guess,bounds=([0.,-2*M,-2*M],[.985*M,8*M,8*M]),
                      xtol=2e-12,ftol=2e-12,gtol=2e-12,diff_step=1e-5,max_nfev=120)
    residual=float(np.max(np.abs(equations(sol.x))))
    if not sol.success or residual>2e-9:
        raise ArithmeticError(f'State solve failed: {sol.message}; residual={residual}')
    phi,nun,nup=sol.x; mass=M-phi
    return _assemble(nB,T,Yp,phi,gas(tfm,nun,mass,rtol=rtol,tail=tail),
                     gas(tfm,nup,mass,rtol=rtol,tail=tail),params,residual)


def fixed_density_gas(T_mev,n,mass,*,rtol=2e-11):
    """Separate scalar bracketing route for stationarity checks (fm units)."""
    tfm=T_mev/HBARC
    upper=max(math.hypot(kf(n),mass)+20*tfm,1.)
    while gas(tfm,upper,mass,rtol=rtol)['n']<n: upper*=2
    nu=brentq(lambda x: gas(tfm,x,mass,rtol=rtol)['n']/n-1,0.,upper,xtol=1e-13,rtol=1e-13)
    return gas(tfm,nu,mass,rtol=rtol)


def sommerfeld_thermal(nB,T,Yp,params=GM1):
    """Leading T^2 free-energy approximation on the cold local RMF branch.

    Reference-density derivatives are central finite differences. This is a
    controlled low-T asymptotic product, not a broad empirical thermal-index fit.
    Reject unless T < .03 of each cold species' kinetic Fermi energy.
    """
    cold=cold_state(nB,Yp,params)
    mass=cold.dirac_mass_mev/HBARC
    energies=[math.hypot(kf(y*nB),mass) for y in (1-Yp,Yp)]
    kinetic_min=min(e-mass for e in energies)*HBARC
    if T<0 or T>.03*kinetic_min:
        raise ValueError('Outside intentionally narrow Sommerfeld domain')
    def coefficient(n):
        s=cold_state(n,Yp,params);m=s.dirac_mass_mev/HBARC
        return sum(kf(y*n)*math.hypot(kf(y*n),m) for y in (1-Yp,Yp))/(3*HBARC)
    gamma=coefficient(nB)
    h=1e-4*nB
    dgamma=(coefficient(nB+h)-coefficient(nB-h))/(2*h)
    return dict(delta_free_energy_density=-.5*gamma*T*T,
                delta_energy_density=.5*gamma*T*T,
                delta_pressure=.5*(gamma-nB*dgamma)*T*T,
                entropy_density=gamma*T,cv_fixed_nB_Yp=gamma*T,
                thermal_index=2-nB*dgamma/gamma,
                min_kinetic_fermi_energy=kinetic_min,
                temperature_over_fermi_scale=T/kinetic_min)
