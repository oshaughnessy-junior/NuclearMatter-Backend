"""Local two-charge stability of frozen GM1 uniform matter.

No coexistence construction or lepton/Coulomb contribution. Outputs chemical
Hessian dmu_i/dn_j|T in MeV fm^3 for order(neutron,proton).
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rmf_reference'))
from gm1_finite_t import state, cold_state, GM1, HBARC


def gas_response(T,nu,mass,*,rtol=2e-11):
    """Analytic distribution derivatives at fixed kinetic nu or m (fm units).

    Return n_nu, rho_s_nu=-n_m, and rho_s_m|nu. Includes thermal antiparticles.
    """
    if not all(math.isfinite(x) for x in (T,nu,mass)) or T<=0 or mass<=0:
        raise ValueError('Require finite T>0, finite nu and mass>0')
    end=max(0.,abs(nu)-mass)+60*T
    panels=sorted(set([0.,end]+[k for k in (abs(nu)-mass-60*T,abs(nu)-mass,abs(nu)-mass+60*T) if 0<k<end]))
    def integrand(k):
        E=mass+k;p=math.sqrt(k*(k+2*mass));jac=p*E/math.pi**2
        x=(E-nu)/T;y=(E+nu)/T
        a,b=expit(-x),expit(-y)
        wa,wb=a*expit(x),b*expit(y)
        return jac*np.array([(wa+wb)/T,(mass/E)*(wa-wb)/T,
                             p*p/(E**3)*(a+b)-(mass/E)**2*(wa+wb)/T])
    result=np.zeros(3)
    for lo,hi in zip(panels[:-1],panels[1:]):
        value,error,info=quad_vec(integrand,lo,hi,epsabs=1e-15,epsrel=rtol,limit=300,full_output=True)
        if not info.success and not (info.status==2 and error <= max(1e-15,rtol*np.max(np.abs(value)))):
            raise ArithmeticError(f'Response quadrature: {info.message}')
        result+=value
    nnu,rsnu,rsmu=result
    if nnu<=0:raise ArithmeticError('Number susceptibility underflow')
    return dict(n_nu=float(nnu),rho_s_nu=float(rsnu),rho_s_m_fixed_nu=float(rsmu),
                rho_s_m_fixed_n=float(rsmu+rsnu*rsnu/nnu),a=float(rsnu/nnu))


def implicit_hessian(s):
    """Implicit scalar-field and number-constraint differentiation on local branch."""
    if s.temperature<=0:raise ValueError('Finite-T response requires T>0')
    mass=s.dirac_mass_mev/HBARC; phi=s.phi_mev/HBARC;M=GM1['M_mev']/HBARC
    n=gas_response(s.temperature/HBARC,s.kinetic_mu_n_mev/HBARC,mass)
    p=gas_response(s.temperature/HBARC,s.kinetic_mu_p_mev/HBARC,mass)
    upp=1/GM1['C_sigma']+2*GM1['b']*M*phi+3*GM1['c']*phi*phi
    D=upp+n['rho_s_m_fixed_n']+p['rho_s_m_fixed_n']
    if abs(D)<1e-10:raise ArithmeticError('Near-singular canonical scalar response')
    a=np.array([n['a'],p['a']])
    H=np.diag([1/n['n_nu'],1/p['n_nu']])-np.outer(a,a)/D
    co=GM1['C_omega']; cr=GM1['C_rho']/4
    H+=np.array([[co+cr,co-cr],[co-cr,co+cr]])
    return H*HBARC,dict(canonical_scalar_curvature_fm_minus2=D,
                        scalar_density_response_a=a.tolist(),
                        dphi_dn_fm2=(a/D).tolist(),neutron=n,proton=p)


def fd_hessian(nB,T,Yp,relative_step=1e-3,anchor=None):
    """Central finite differences of chemical potentials at fixed other species."""
    nn=nB*(1-Yp);np_=nB*Yp
    H=np.zeros((2,2))
    for j,base in enumerate((nn,np_)):
        h=relative_step*base
        nm=np.array([nn,np_]);npair=nm.copy();nm[j]-=h;npair[j]+=h
        def mu(pair):
            s=state(float(sum(pair)),T,float(pair[1]/sum(pair)),initial=anchor)
            return np.array([s.mu_n_mev,s.mu_p_mev])
        H[:,j]=(mu(npair)-mu(nm))/(2*h)
    return H


def classify(s,H,details=None):
    """Local stability with species densities conserved, in nuclear-only matter."""
    H=np.asarray(H,dtype=float)
    if H.shape!=(2,2) or not np.all(np.isfinite(H)):raise ValueError('Finite2x2 Hessian required')
    symmetric=(H+H.T)/2
    eigenvalues,eigenvectors=np.linalg.eigh(symmetric)
    v=np.array([1-s.Yp,s.Yp]);u=np.array([-1.,1.])
    A=float(v@symmetric@v); B=float(v@symmetric@u);C=float(u@symmetric@u)
    condition_scale=max(1.,float(np.linalg.norm(symmetric,ord=2)))
    tol=1e-7*condition_scale
    label='locally_stable' if eigenvalues[0]>tol else ('spinodal' if eigenvalues[0]<-tol else 'marginal_unqualified')
    symmetry_error=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H),1.))
    if symmetry_error>1e-6:label='unqualified_nonsymmetric_hessian'
    if details is not None and details['canonical_scalar_curvature_fm_minus2']<=0:
        label='scalar_unstable_branch'
    return dict(nB=s.nB,T=s.temperature,Yp=s.Yp,hessian_mev_fm3=H.tolist(),
                antisymmetric_scaled_norm=symmetry_error,
                eigenvalues_mev_fm3=eigenvalues.tolist(),unstable_eigenvector_n_p=eigenvectors[:,0].tolist(),
                mechanical_dP_dn_fixed_T_Yp=s.nB*A,
                composition_dmu_difference_dYp_fixed_n=s.nB*C,
                diffusive_dmu_difference_dYp_fixed_pressure=s.nB*(C-B*B/A) if abs(A)>tol else None,
                local_label=label,
                fixed_Y_mechanically_stable=bool(A>tol),
                composition_at_fixed_n_convex=bool(C>tol),
                globally_stable_phase='unassessed; no two-charge coexistence/Coulomb/lepton model',
                implicit_response=details)
