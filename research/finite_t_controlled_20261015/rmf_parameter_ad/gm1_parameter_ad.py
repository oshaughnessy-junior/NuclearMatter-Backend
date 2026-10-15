"""First-order implicit coupling sensitivities of the frozen finite-T RMF oracle.

Raw input order=(nB,T_MeV,Yp,C_sigma,C_omega,C_rho,b,c), fixed bare M=939MeV.
No EOS-family fit, emulator, higher-order derivative or production claim.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
for directory in ('rmf_reference','rmf_phase','rmf_ad'):
    sys.path.insert(0,str(ROOT/directory))
from gm1_finite_t import state,GM1,HBARC
from stability import gas_response
from gm1_implicit_ad import thermal_response,OUTPUTS,NOUT

PARAMETERS=('C_sigma','C_omega','C_rho','b','c')
INPUTS=('nB','T_MeV','Yp')+PARAMETERS
NINPUT=8


def pack_state(nB,T,Yp,params=None):
    if params is not None and set(params)-set(PARAMETERS):
        raise ValueError('Only five named coupling overrides supported; bare M is fixed939MeV')
    p=dict(GM1);p.update(params or {})
    return np.array([nB,T,Yp]+[p[k] for k in PARAMETERS],dtype=np.float64)


def unpack(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(8,) or not np.all(np.isfinite(x)) or x[0]<=0 or x[1]<=0 or not 0<x[2]<1 or np.any(x[3:6]<=0):
        raise ValueError('Require finite raw8-vector, nB,T,C_sigma/C_omega/C_rho>0,0<Yp<1')
    p=dict(GM1);p.update(zip(PARAMETERS,map(float,x[3:])))
    return tuple(map(float,x[:3])),p


def oracle_value_jacobian(x,*,rtol=2e-11,tail=60.,return_diagnostics=False):
    (nB,T,Y),p=unpack(x)
    s=state(nB,T,Y,params=p,rtol=rtol,tail=tail)
    phi=s.phi_mev/HBARC;m=s.dirac_mass_mev/HBARC;M=939./HBARC;tf=T/HBARC
    nu=np.array([s.kinetic_mu_n_mev,s.kinetic_mu_p_mev])/HBARC
    responses=[gas_response(tf,float(v),m,rtol=rtol) for v in nu]
    thermal=[thermal_response(tf,float(v),m,rtol=rtol) for v in nu]
    chi=np.array([r['n_nu'] for r in responses]);b=np.array([r['rho_s_nu'] for r in responses]);q=np.array([r['rho_s_m_fixed_nu'] for r in responses])
    nt=np.array([t['n_T'] for t in thermal]);rt=np.array([t['rho_s_T'] for t in thermal]);st=np.array([t['s_T'] for t in thermal])
    upp=1/p['C_sigma']+2*p['b']*M*phi+3*p['c']*phi*phi
    Fz=np.array([[upp+sum(q),-b[0],-b[1]],[b[0],chi[0],0.],[b[1],0.,chi[1]]])
    D=upp+sum(q+b*b/chi)
    if D<=1e-10 or np.linalg.cond(Fz)>1e10:
        raise ArithmeticError('Unqualified scalar/number implicit Jacobian')
    dn=np.zeros((2,NINPUT));dn[:,:3]=np.array([[1-Y,0.,-nB],[Y,0.,nB]])
    dt=np.zeros(NINPUT);dt[1]=1/HBARC
    Fx=np.vstack([-sum(rt)*dt,nt[0]*dt-dn[0],nt[1]*dt-dn[1]])
    # Explicit scalar-equation coupling terms at fixed Phi,nu,T.
    Fx[0,3:]=np.array([-phi/(p['C_sigma']**2),0.,0.,M*phi*phi,phi**3])
    dz=np.linalg.solve(Fz,-Fx);dphi,dnun,dnup=dz
    dW=p['C_omega']*(dn[0]+dn[1]);dW[4]+=nB
    n3=nB*(2*Y-1)
    dR=.5*p['C_rho']*(dn[1]-dn[0]);dR[5]+=.5*n3
    dmun=(dnun+dW-.5*dR)*HBARC
    dmup=(dnup+dW+.5*dR)*HBARC
    ds=sum(st)*dt+nt[0]*dnun+nt[1]*dnup+sum(rt)*dphi
    # Reduced canonical free energy after algebraically eliminating vectors;
    # scalar stationarity removes implicit field terms (Hellmann-Feynman).
    hf=HBARC*np.array([-phi*phi/(2*p['C_sigma']**2),nB*nB/2,n3*n3/8,M*phi**3/3,phi**4/4])
    parameter_work=np.zeros(NINPUT);parameter_work[3:]=hf
    dt_mev=np.zeros(NINPUT);dt_mev[1]=1.
    df=-s.entropy_density*dt_mev+s.mu_n_mev*dn[0]+s.mu_p_mev*dn[1]+parameter_work
    de=T*ds+s.mu_n_mev*dn[0]+s.mu_p_mev*dn[1]+parameter_work
    # Parameter work must be subtracted from Gibbs-Duhem at varying couplings.
    dP=s.entropy_density*dt_mev+nB*(1-Y)*dmun+nB*Y*dmup-parameter_work
    J=np.vstack([dphi*HBARC,dW*HBARC,dR*HBARC,-dphi*HBARC,
                 dnun*HBARC,dnup*HBARC,dmun,dmup,de,dP,df,ds])
    y=np.array([getattr(s,k) for k in OUTPUTS],dtype=np.float64)
    if not np.all(np.isfinite(J)):raise ArithmeticError('Nonfinite coupling Jacobian')
    if return_diagnostics:
        a=b/chi
        H=np.diag(1/chi)-np.outer(a,a)/D
        co=p['C_omega'];cr=p['C_rho']/4
        H+=np.array([[co+cr,co-cr],[co-cr,co+cr]])
        return y,J,dict(params=p,max_equation_residual=s.max_normalized_equation_residual,
                        canonical_scalar_curvature_fm_minus2=float(D),
                        implicit_jacobian_condition=float(np.linalg.cond(Fz)),
                        implicit_linear_residual=float(np.max(np.abs(Fz@dz+Fx))),
                        chemical_hessian_eigenvalues_mev_fm3=np.linalg.eigvalsh(H*HBARC).tolist(),
                        hellmann_feynman_free_energy_derivatives=hf.tolist(),
                        naive_fixed_parameter_pressure_derivatives=(nB*(1-Y)*dmun+nB*Y*dmup)[3:].tolist())
    return y,J


def _value(x):
    try:
        (n,T,Y),p=unpack(x);s=state(n,T,Y,params=p)
        return np.array([getattr(s,k) for k in OUTPUTS],dtype=np.float64)
    except (ValueError,ArithmeticError,np.linalg.LinAlgError,RuntimeError):
        return np.full(NOUT,np.nan,dtype=np.float64)


def _value_jac(x):
    try:return oracle_value_jacobian(x)
    except (ValueError,ArithmeticError,np.linalg.LinAlgError,RuntimeError):
        return np.full(NOUT,np.nan,dtype=np.float64),np.full((NOUT,NINPUT),np.nan,dtype=np.float64)


@jax.custom_jvp
def state_parameter_ad(x):
    """First-order8-input oracle, fixed M939; raw C couplings in fm²,b/c dimensionless."""
    x=jnp.asarray(x,dtype=jnp.float64)
    if x.shape!=(NINPUT,):raise ValueError('Expected raw shape(8,)')
    return jax.pure_callback(_value,jax.ShapeDtypeStruct((NOUT,),jnp.float64),x,vmap_method='sequential')


@state_parameter_ad.defjvp
def _jvp(primals,tangents):
    (x,),(dx,)=primals,tangents
    y,J=jax.pure_callback(_value_jac,(jax.ShapeDtypeStruct((NOUT,),jnp.float64),jax.ShapeDtypeStruct((NOUT,NINPUT),jnp.float64)),x,vmap_method='sequential')
    return y,J@dx
