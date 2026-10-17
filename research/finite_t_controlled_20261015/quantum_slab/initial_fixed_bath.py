"""Selected-energy FD-nucleon fixed-bath plane slab, one helicity per flavor.

Named LO allowed-current NR capture/inverse and crossed-neutron-decay/inverse.
No scattering, redistribution, fluid evolution, interactions or full spectrum.
"""
from __future__ import annotations
from pathlib import Path
import sys
import math
import numpy as np
from scipy.special import expit,expn
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for directory in ('nr_baryon_eos','capture_endpoint','crossed_opacity','allowed_opacity','weak_rates'):
    sys.path.insert(0,str(ROOT/directory))
from free_nucleons import matter_state
import mapped_capture
import guarded_capture
import crossed_decay
import allowed_capture

LIGHT=2.99792458e10
HBARC=197.3269804
DOS=1/(2*math.pi**2*HBARC**3)  # one massless neutrino helicity


def coefficients(matter,flavor,energy,*,order=48,tail=40.):
    """Independently calculated j,chi for the named channel inventory, s^-1."""
    controls=dict(energy_order=order,angle_order=order,tail=tail)
    if flavor not in ('nue','antinue'):raise ValueError('Expected nue/antinue')
    optical=matter.optical(flavor)
    try:
        capture=mapped_capture.log_coefficients(optical,energy,**controls)
        guard=dict(used=False,discarded_panels=[])
    except ValueError as exc:
        lj_guard,lc_guard,discarded=guarded_capture.log_coefficients(optical,energy,**controls)
        if not discarded:raise
        capture=(lj_guard,lc_guard)
        guard=dict(used=True,original_failure=str(exc),discarded_panels=discarded)
    channels={'capture':dict(log_j=capture[0],log_chi=capture[1])}
    lj,lc=capture
    if flavor=='antinue':
        decay=crossed_decay.DecayState(matter.pair('neutron'),matter.electron.chemical_potential)
        cross=crossed_decay.log_coefficients(decay,energy,**controls)
        channels['crossed_neutron_decay']=dict(log_j=cross[0],log_chi=cross[1])
        lj=float(np.logaddexp(lj,cross[0]));lc=float(np.logaddexp(lc,cross[1]))
    log_rate=float(np.logaddexp(lj,lc))
    j=math.exp(lj);chi=math.exp(lc);rate=math.exp(log_rate)
    if not all(math.isfinite(v) and v>0 for v in (j,chi,rate)):
        raise ArithmeticError('Selected coefficient underflow/nonfinite; log transfer extension needed')
    feq=float(expit(lj-lc))
    return dict(flavor=flavor,energy_MeV=energy,log_j=lj,log_chi=lc,j_s= j,chi_s=chi,
                rate_s=rate,feq=feq,mu_neutrino_MeV=optical.mu_neutrino,
                true_absorption_cm_inv=chi/LIGHT,relaxation_cm_inv=rate/LIGHT,
                true_over_relaxation=chi/rate,channels=channels,capture_panel_guard=guard)


def boundary_occupations(energy):
    """Prescribed isotropic incoming hemispheres; deliberately different baths."""
    return float(expit((3.-energy)/4.)),float(.35*expit((-2.-energy)/6.))


def ray_occupation(t,mu,tau,feq,left,right):
    """Optical coordinate t=rate*x/c, signed direction cosine mu,0<=t<=tau."""
    if mu==0 or not -1<=mu<=1 or not 0<=t<=tau:raise ValueError('Invalid ray coordinates')
    distance=t if mu>0 else tau-t
    incident=left if mu>0 else right
    attenuation=math.exp(-distance/abs(mu))
    return feq*(-math.expm1(-distance/abs(mu)))+incident*attenuation


def fluxes(coeff,length_cm,left,right):
    """Number flux per energy, cm^-2 s^-1 MeV^-1; positive outward each face.

    Incoming values are one outward-magnitude incoming hemisphere per face.
    Radiation gain=both outgoing magnitudes minus both incoming magnitudes.
    """
    if length_cm<=0 or not all(0<=v<=1 for v in (left,right)):raise ValueError('Invalid slab/boundary')
    E=coeff['energy_MeV'];eq=coeff['feq'];tau=coeff['relaxation_cm_inv']*length_cm
    A=LIGHT*DOS*E*E*1e39
    escape=.5-float(expn(3,tau))
    # Stable thin-limit expression avoids subtracting two large total fluxes.
    transmission=float(expn(3,tau))
    outgoing_left=A/2*(eq*escape+right*transmission)
    outgoing_right=A/2*(eq*escape+left*transmission)
    incoming_left=A/4*left;incoming_right=A/4*right
    net=A*(eq-.5*(left+right))*escape
    return dict(length_cm=length_cm,tau_relaxation=tau,
                tau_true_absorption=coeff['true_absorption_cm_inv']*length_cm,
                incoming_left=incoming_left,incoming_right=incoming_right,
                outgoing_left=outgoing_left,outgoing_right=outgoing_right,
                outgoing_both=outgoing_left+outgoing_right,
                radiation_number_gain=net,radiation_energy_gain_MeV=E*net,
                radiation_electron_lepton_gain=(1 if coeff['flavor']=='nue' else -1)*net,
                bath_electron_lepton_source=-(1 if coeff['flavor']=='nue' else -1)*net,
                bath_number_source=-net,bath_energy_source_MeV=-E*net)


def numerical_boundary_flux(coeff,tau,left,right):
    """Independent angular QUADPACK integral of the analytic ray solution."""
    eq=coeff['feq'];A=LIGHT*DOS*coeff['energy_MeV']**2*1e39
    def integrate_face(incoming,side):
        scale=max(eq*(-math.expm1(-tau))+incoming*math.exp(-tau),1e-300)
        fn=(lambda mu:mu*ray_occupation(0.,-mu,tau,eq,left,right)) if side=='left' else (lambda mu:mu*ray_occupation(tau,mu,tau,eq,left,right))
        return quad(fn,0.,1.,epsabs=1e-12*scale,epsrel=1e-10,limit=300)[0]
    outgoing_l=integrate_face(right,'left')
    outgoing_r=integrate_face(left,'right')
    return A*.5*outgoing_l,A*.5*outgoing_r


def numerical_volume_gain(coeff,tau,left,right):
    """Independent optical-depth QUADPACK integral of angular-averaged C.

    Each illuminated surface is integrated inward on explicit optical-depth
    panels, resolving the opaque boundary layer rather than missing it.
    """
    eq=coeff['feq'];A=LIGHT*DOS*coeff['energy_MeV']**2*1e39
    edges=sorted(set([0.,tau]+[v for v in (.01,.1,1.,3.,10.,30.,100.) if v<tau]))
    integral=sum(quad(lambda t:float(expn(2,t)),lo,hi,epsabs=1e-12,epsrel=1e-10,limit=200)[0] for lo,hi in zip(edges[:-1],edges[1:]))
    return A*.5*(2*eq-left-right)*integral
