"""First-order JAX interface to the exact CPU GM1 state oracle.

Inputs x=(nB [fm^-3], T [MeV], Yp); outputs documented by OUTPUTS.
Primal values use the frozen nonlinear root/quadrature reference. JVP/VJP use
analytic distribution moments and the exact implicit3x3constraint Jacobian.
No finite-difference derivative rule, learned emulator or second-order claim.
"""
from __future__ import annotations
from pathlib import Path
import sys
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rmf_reference'))
sys.path.insert(0,str(ROOT/'rmf_phase'))
from gm1_finite_t import state,GM1,HBARC
from stability import gas_response

OUTPUTS=('phi_mev','omega_mev','rho_mev','dirac_mass_mev',
         'kinetic_mu_n_mev','kinetic_mu_p_mev','mu_n_mev','mu_p_mev',
         'energy_density','pressure','free_energy_density','entropy_density')
NOUT=len(OUTPUTS)


def thermal_response(T,nu,mass,*,rtol=2e-11):
    """n_T|nu,m, rho_s_T|nu,m, s_T|nu,m; all internal fm units."""
    end=max(0.,abs(nu)-mass)+60*T
    panels=sorted(set([0.,end]+[k for k in (abs(nu)-mass-60*T,abs(nu)-mass,abs(nu)-mass+60*T) if 0<k<end]))
    def integrand(k):
        E=mass+k;p=math.sqrt(k*(k+2*mass));jac=p*E/math.pi**2
        x=(E-nu)/T;y=(E+nu)/T
        wa=expit(-x)*expit(x);wb=expit(-y)*expit(y)
        return jac*np.array([((E-nu)*wa-(E+nu)*wb)/(T*T),
                             mass/E*((E-nu)*wa+(E+nu)*wb)/(T*T),
                             ((E-nu)**2*wa+(E+nu)**2*wb)/(T**3)])
    out=np.zeros(3)
    for lo,hi in zip(panels[:-1],panels[1:]):
        val,err,info=quad_vec(integrand,lo,hi,epsabs=1e-15,epsrel=rtol,limit=300,full_output=True)
        if not info.success and not (info.status==2 and err<=max(1e-15,rtol*np.max(np.abs(val)))):
            raise ArithmeticError(f'Thermal response quadrature: {info.message}')
        out+=val
    return dict(n_T=float(out[0]),rho_s_T=float(out[1]),s_T=float(out[2]))


def oracle_value_jacobian(x,*,rtol=2e-11,tail=60.,return_diagnostics=False):
    """Exact primal oracle and analytic first Jacobian, with no FD derivatives."""
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(3,) or not np.all(np.isfinite(x)) or x[1]<=0:
        raise ValueError('Require finite3-vector(nB,T>0,Yp)')
    nB,T,Y=map(float,x)
    s=state(nB,T,Y,rtol=rtol,tail=tail)
    m=s.dirac_mass_mev/HBARC;phi=s.phi_mev/HBARC;M=GM1['M_mev']/HBARC
    tf=T/HBARC
    nu=np.array([s.kinetic_mu_n_mev,s.kinetic_mu_p_mev])/HBARC
    r=[gas_response(tf,float(v),m,rtol=rtol) for v in nu]
    t=[thermal_response(tf,float(v),m,rtol=rtol) for v in nu]
    chi=np.array([v['n_nu'] for v in r]);b=np.array([v['rho_s_nu'] for v in r]);q=np.array([v['rho_s_m_fixed_nu'] for v in r])
    nt=np.array([v['n_T'] for v in t]);rt=np.array([v['rho_s_T'] for v in t])
    st=np.array([v['s_T'] for v in t])
    upp=1/GM1['C_sigma']+2*GM1['b']*M*phi+3*GM1['c']*phi*phi
    # z=(Phi,nu_n,nu_p); F=(scalar gap,neutron number,proton number).
    Fz=np.array([[upp+sum(q),-b[0],-b[1]],
                 [b[0],chi[0],0.],[b[1],0.,chi[1]]])
    canonical=upp+sum(q+b*b/chi)
    if canonical<=1e-10 or np.linalg.cond(Fz)>1e10:
        raise ArithmeticError('Unqualified scalar/number implicit Jacobian')
    dn=np.array([[1-Y,0.,-nB],[Y,0.,nB]])
    dT=np.array([0.,1/HBARC,0.])
    Fx=np.vstack([-sum(rt)*dT,nt[0]*dT-dn[0],nt[1]*dT-dn[1]])
    dz=np.linalg.solve(Fz,-Fx)
    dphi,dnun,dnup=dz
    dW=GM1['C_omega']*np.array([1.,0.,0.])
    dR=.5*GM1['C_rho']*np.array([2*Y-1,0.,2*nB])
    dmun=(dnun+dW-.5*dR)*HBARC
    dmup=(dnup+dW+.5*dR)*HBARC
    # Maxwell identities of the ideal sectors: s_nu=n_T, s_m=-rho_s_T.
    ds=sum(st)*dT+nt[0]*dnun+nt[1]*dnup+sum(rt)*dphi
    dT_mev=np.array([0.,1.,0.])
    df=-s.entropy_density*dT_mev+s.mu_n_mev*dn[0]+s.mu_p_mev*dn[1]
    de=T*ds+s.mu_n_mev*dn[0]+s.mu_p_mev*dn[1]
    dP=s.entropy_density*dT_mev+nB*(1-Y)*dmun+nB*Y*dmup
    J=np.vstack([dphi*HBARC,dW*HBARC,dR*HBARC,-dphi*HBARC,
                 dnun*HBARC,dnup*HBARC,dmun,dmup,de,dP,df,ds])
    y=np.array([getattr(s,name) for name in OUTPUTS],dtype=np.float64)
    if not np.all(np.isfinite(J)):raise ArithmeticError('Non-finite analytic Jacobian')
    if return_diagnostics:
        return y,J,dict(max_equation_residual=s.max_normalized_equation_residual,
                        implicit_jacobian_condition=float(np.linalg.cond(Fz)),
                        canonical_scalar_curvature_fm_minus2=float(canonical),
                        implicit_linear_residual=float(np.max(np.abs(Fz@dz+Fx))),
                        canonical_cv_formula=float(tf*(sum(st-nt*nt/chi)+(sum(rt-b*nt/chi))**2/canonical)),
                        canonical_dphi_dT_mev_per_mev=float(sum(rt-b*nt/chi)/canonical))
    return y,J


def _value(x):
    # Runtime exceptions are side effects for pure_callback. Numerical/domain
    # failures return a deterministic NaN sentinel; use the strict host oracle
    # for preflight diagnostics instead of relying on callback exceptions.
    try:
        n,T,Y=map(float,np.asarray(x))
        if T<=0:raise ValueError('AD reference requires T>0')
        s=state(n,T,Y)
        return np.array([getattr(s,name) for name in OUTPUTS],dtype=np.float64)
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,RuntimeError):
        return np.full(NOUT,np.nan,dtype=np.float64)


def _value_jac(x):
    try:
        y,J=oracle_value_jacobian(x)
        return y,J
    except (ArithmeticError,ValueError,np.linalg.LinAlgError,RuntimeError):
        return np.full(NOUT,np.nan,dtype=np.float64),np.full((NOUT,3),np.nan,dtype=np.float64)


@jax.custom_jvp
def state_ad(x):
    """First-order differentiable exact state oracle; CPU callbacks, float64.

    Qualified only for scalar3-input state calls and first JVP/VJP/jacfwd/jacrev.
    Higher derivatives through the callback deliberately remain unsupported.
    """
    x=jnp.asarray(x,dtype=jnp.float64)
    if x.shape!=(3,):raise ValueError('Expected shape(3,)')
    return jax.pure_callback(_value,jax.ShapeDtypeStruct((NOUT,),jnp.float64),x,vmap_method='sequential')


@state_ad.defjvp
def _state_ad_jvp(primals,tangents):
    (x,),(dx,)=primals,tangents
    y,J=jax.pure_callback(_value_jac,
        (jax.ShapeDtypeStruct((NOUT,),jnp.float64),jax.ShapeDtypeStruct((NOUT,3),jnp.float64)),
        x,vmap_method='sequential')
    return y,J@dx
