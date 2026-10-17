"""LO free-nucleon neutral-current scattering with NR FD phase space.

Kernel d²Gamma/(dEout dcos) [s^-1 MeV^-1], before final neutrino blocking.
Density and per-component Pauli-spin responses coincide ONLY in this free,
unpolarized model. No CKM factor, weak magnetism, correlations or RPA.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import math
import numpy as np
from scipy.integrate import quad_vec
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for d in ('nr_baryon_eos','response_reference','weak_rates'):sys.path.insert(0,str(ROOT/d))
from free_nucleons import state_density
from free_density_response import PairState,log_response
from static_capture import GF,GA,HBAR,HBARC,ME

SIN2_THETA_W=.23129  # declared tree-level benchmark, no running/strange correction
LIGHT=2.99792458e10
K=GF*GF*HBARC**3/(4*math.pi**2*HBAR)
DOS=1/(2*math.pi**2*HBARC**3)  # one neutrino helicity


@dataclass(frozen=True)
class Couplings:
    vector:float
    axial:float
    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.vector,self.axial)) or self.vector==self.axial==0:
            raise ValueError('Finite nonzero NC couplings required')


NEUTRON=Couplings(-.5,-GA/2)
PROTON=Couplings(.5-2*SIN2_THETA_W,GA/2)


def pair_from_species(species):
    return PairState(species.T,species.mass,species.offset,species.offset,species.mu,species.mu)


def angular_weight(c,couplings):
    return couplings.vector**2*(1+c)+couplings.axial**2*(3-c)


def _angle_contract(angle_order):
    if isinstance(angle_order,bool) or not isinstance(angle_order,(int,np.integer)) or angle_order<2:
        raise ValueError('Require integer angle_order>=2')


def log_kernel(pair,Ein,Eout,cosine,couplings=NEUTRON):
    """Bare outscattering kernel, no incoming f or outgoing(1-f) included.

    omega=Ein-Eout is transferred INTO nucleons. q=0 is distributional;
    angular quadrature excludes the exact forward/elastic singular point.
    """
    if pair.offset_a!=pair.offset_b or pair.mu_a!=pair.mu_b:
        raise ValueError('NC requires diagonal same-species response')
    if not all(math.isfinite(x) for x in (Ein,Eout,cosine)) or Ein<=0 or Eout<=0 or not -1<=cosine<=1:
        raise ValueError('Require positive finite neutrino energies and cosine in[-1,1]')
    q=math.sqrt((Ein-Eout)**2+2*Ein*Eout*(1-cosine))
    w=angular_weight(cosine,couplings)
    if w==0:return -math.inf
    if q==0:raise ValueError('q=0 is distributional')
    return math.log(K)+2*math.log(Eout)+math.log(w)+log_response(pair,q,Ein-Eout)


def kernel(pair,Ein,Eout,cosine,couplings=NEUTRON,final_occupation=0.):
    """Outscattering per occupied incident neutrino, final Pauli factor explicit."""
    if not math.isfinite(final_occupation) or not 0<=final_occupation<=1:raise ValueError('Final occupation outside[0,1]')
    return math.exp(log_kernel(pair,Ein,Eout,cosine,couplings))*(1-final_occupation)


def moments(species,Ein,couplings=NEUTRON,*,angle_order=32,energy_rtol=2e-9,tail=40.,radiation=None):
    """Selected Ein: zeroth,1-cos and omega moments; positive Eout integration.

    radiation=None gives unblocked event moments. Otherwise(Tnu,mu_nu)
    specifies final neutrino FD blocking only; the incident occupancy is absent.
    The1-cos moment is an angular transport diagnostic, not a full inelastic
    radiation transport closure.
    """
    _angle_contract(angle_order)
    if not all(math.isfinite(x) and x>0 for x in (Ein,energy_rtol,tail)):
        raise ValueError('Require finite positive Ein, energy_rtol and tail')
    if radiation is not None:
        if len(radiation)!=2 or not all(math.isfinite(x) for x in radiation) or radiation[0]<=0:
            raise ValueError('Require radiation=(finite positive Tnu, finite mu_nu)')
    pair=pair_from_species(species)
    cs,cws=leggauss(angle_order)
    total=np.zeros(3)
    for c,cw in zip(cs,cws):
        q0=Ein*math.sqrt(2*(1-c));recoil=q0*q0/(2*pair.mass)
        width=max(q0*math.sqrt(pair.T/pair.mass),q0*math.sqrt(2*max(species.kinetic_mu,0.)/pair.mass),recoil,1e-12)
        center=Ein-recoil
        upper=Ein+max(tail*pair.T,24*width)
        edges=sorted(set([0.,upper,Ein]+[x for x in [center+k*width for k in(-24,-12,-6,-3,-1,0,1,3,6,12,24)] if 0<x<upper]))
        def integrand(Eout):
            if Eout<=0:return np.zeros(3)
            f=0. if radiation is None else float(expit((radiation[1]-Eout)/radiation[0]))
            value=kernel(pair,Ein,Eout,float(c),couplings,f)
            return value*np.array([1.,1-c,Ein-Eout])
        for lo,hi in zip(edges[:-1],edges[1:]):
            val,err,info=quad_vec(integrand,lo,hi,epsabs=1e-18,epsrel=energy_rtol,limit=250,full_output=True)
            if not info.success and not (info.status==2 and err<=max(1e-18,energy_rtol*np.max(np.abs(val)))):
                raise ArithmeticError(f'Energy quadrature: {info.message}')
            total+=cw*val
    return dict(rate_s=float(total[0]),angular_transport_rate_s=float(total[1]),energy_to_bath_MeV_s=float(total[2]),
                inverse_mfp_cm_inv=float(total[0]/LIGHT),angular_transport_inverse_mfp_cm_inv=float(total[1]/LIGHT))


def bare_static_cross_sections(E,couplings=NEUTRON):
    """Independent angle-integrated static LO formulas, cm² per target."""
    cv,ca=couplings.vector,couplings.axial
    pref=GF*GF*E*E*HBARC**2*1e-26/math.pi
    return dict(total_cm2=pref*(cv*cv+3*ca*ca),transport_cm2=2*pref/3*(cv*cv+5*ca*ca))


def discrete_collision(species,energies,weights,occupations,couplings=NEUTRON,angle_order=48):
    """Closed selected grid: number-conservative pairwise scattering, no escape.

    This deliberately omits transitions outside the grid; it is a reversible
    quadrature structure check, not a converged continuum-time-evolution model.
    """
    _angle_contract(angle_order)
    E=np.asarray(energies,float);w=np.asarray(weights,float);f=np.asarray(occupations,float)
    if E.ndim!=1 or E.size==0 or w.shape!=E.shape or f.shape!=E.shape or not all(np.all(np.isfinite(x)) for x in (E,w,f)) or np.any(E<=0) or np.any(w<=0) or np.any((f<0)|(f>1)):
        raise ValueError('Invalid positive energy-grid/occupation arrays')
    cs,cws=leggauss(angle_order);pair=pair_from_species(species)
    A=np.zeros((len(E),len(E)))
    for i,Ein in enumerate(E):
        for j,Eout in enumerate(E):
            A[i,j]=w[j]*sum(cw*kernel(pair,float(Ein),float(Eout),float(c),couplings) for c,cw in zip(cs,cws))
    g=DOS*E*E*w
    flux=g[:,None]*A*f[:,None]*(1-f[None,:])
    # Diagonal self-transitions cancel identically. Never subtract their large
    # row/column sums when off-diagonal redistribution is much smaller.
    dN=np.zeros(len(E))
    for i in range(len(E)):
        for j in range(i):
            net=flux[j,i]-flux[i,j]
            dN[i]+=net
            dN[j]-=net
    C=dN/g
    return dict(collision=C,number_source=dN,density_weights=g,transition_rates=A,
                net_number_source=float(sum(dN)),net_neutrino_energy_source=float(E@dN))
