"""Crossed neutron-decay optical channel in the LO allowed NR ansatz."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
from scipy.special import log_expit, logsumexp

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"allowed_opacity"))
import allowed_capture as allowed
from free_density_response import PairState, log_response
import static_capture as constants


@dataclass(frozen=True)
class DecayState:
    baryons: PairState  # ordered n->p, i.e. SPONTANEOUS direction
    mu_e: float  # total electron chemical potential in MeV

    def __post_init__(self):
        if not math.isfinite(self.mu_e):
            raise ValueError("Require finite total electron chemical potential")

    @property
    def mu_antineutrino(self):
        return self.baryons.mu_a-self.baryons.mu_b-self.mu_e


def _panels(state,Enu,c,tail):
    s=state.baryons
    Qeff=s.offset_a-s.offset_b
    E0=Qeff-Enu
    Ecentral=max(constants.ME,E0)
    pe=math.sqrt(max(0.,Ecentral*Ecentral-constants.ME*constants.ME))
    q0=math.sqrt(Enu*Enu+pe*pe+2*Enu*pe*c)
    thermal=q0*math.sqrt(s.T/s.mass)
    fermi=q0*math.sqrt(2*max(s.kinetic_mu_a,s.kinetic_mu_b,0.)/s.mass)
    recoil=q0*q0/(2*s.mass)
    width=max(thermal,fermi,recoil,1e-12)
    upper=max(constants.ME,E0,state.mu_e)+tail*s.T+16*width+recoil
    candidates=[constants.ME,upper,E0,E0-recoil]
    candidates.extend(E0-recoil+k*width for k in (-16,-8,-4,-2,-1,1,2,4,8,16))
    candidates.extend(state.mu_e+k*s.T for k in (-12,-4,0,4,12))
    return sorted(set(x for x in candidates if constants.ME<=x<=upper))


def log_coefficients(state: DecayState,Enu,energy_order=24,angle_order=24,tail=40.):
    """Return log j_nd, log chi_inverse3body in s^-1, Enu in MeV.

    Baryon transfer for decay is (-E_e-E_nu, -p_e-p_nu), unlike capture.
    """
    if not math.isfinite(Enu) or Enu<=0:
        raise ValueError("Require finite Enu>0")
    if energy_order<4 or angle_order<4 or not math.isfinite(tail) or tail<=0:
        raise ValueError("Invalid quadrature controls")
    cs,cws=allowed.nodes(angle_order)
    xs,xws=allowed.nodes(energy_order)
    sj,sc=[],[]
    s=state.baryons
    sr=s.reverse()
    for c,cw in zip(cs,cws):
        edges=_panels(state,Enu,float(c),tail)
        for lo,hi in zip(edges[:-1],edges[1:]):
            # Smooth the true finite-electron-mass threshold analytically,
            # rather than asking polynomial quadrature to resolve sqrt(E-m).
            xlo=math.sqrt(lo-constants.ME)
            xhi=math.sqrt(hi-constants.ME)
            xx=xlo+.5*(xs+1)*(xhi-xlo)
            Es=constants.ME+xx*xx
            weights=.5*(xhi-xlo)*xws*2*xx
            for Ee,ew in zip(Es,weights):
                pe=math.sqrt(max(0.,Ee*Ee-constants.ME*constants.ME))
                q=math.sqrt(Enu*Enu+pe*pe+2*Enu*pe*c)
                omega=-Enu-Ee
                common=math.log(float(cw*ew*Ee*pe*allowed.angular_factor(Ee,c)))
                sj.append(common+log_expit((Ee-state.mu_e)/s.T)+log_response(s,q,omega))
                sc.append(common+log_expit((state.mu_e-Ee)/s.T)+log_response(sr,q,-omega))
    return float(math.log(allowed.K)+logsumexp(sj)),float(math.log(allowed.K)+logsumexp(sc))


def coefficients(state,Enu,**kwargs):
    lj,lc=log_coefficients(state,Enu,**kwargs)
    j,chi=math.exp(lj),math.exp(lc)
    return {"j_s-1":j,"chi_s-1":chi,"true_absorption_cm-1":chi/allowed.C_CM_S,
            "relaxation_cm-1":(j+chi)/allowed.C_CM_S}


def combined_antineutrino(decay_state: DecayState,Enu,**kwargs):
    """Positron capture PLUS neutron decay, each with its own inverse.

    Call only after individually qualified constituents. Does not include
    pair/scattering kernels or current corrections beyond the same LO ansatz.
    """
    pc=allowed.OpticalState(decay_state.baryons.reverse(),-decay_state.mu_e)
    lpj,lpc=allowed.log_coefficients(pc,Enu,**kwargs)
    ldj,ldc=log_coefficients(decay_state,Enu,**kwargs)
    return {"log_j_s-1":float(np.logaddexp(lpj,ldj)),
            "log_chi_s-1":float(np.logaddexp(lpc,ldc)),
            "positron_capture_log_j_s-1":lpj,"neutron_decay_log_j_s-1":ldj}
