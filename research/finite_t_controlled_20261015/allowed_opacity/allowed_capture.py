"""Research allowed-current absorption/emission with NR recoil+blocking.

Leading static V0/Aspin current vertices, dynamical common-mass phase space.
No recoil-current, weak-magnetism or relativistic-medium claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit, logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "response_reference"))
sys.path.insert(0, str(ROOT / "weak_rates"))
from free_density_response import PairState, log_response
import static_capture as constants

ME = constants.ME
GV = 1.
GA = constants.GA
K = constants.GF**2*constants.VUD**2*constants.HBARC**3/(4*math.pi**2*constants.HBAR)
C_CM_S = 2.99792458e10


@dataclass(frozen=True)
class OpticalState:
    baryons: PairState  # ordered species a->b for neutrino absorption
    mu_lepton: float  # mu_e for nu_e+n; -mu_e for antinu_e+p

    def __post_init__(self):
        if not math.isfinite(self.mu_lepton):
            raise ValueError("Require finite total charged-lepton chemical potential")

    @property
    def mu_neutrino(self):
        s=self.baryons
        return self.mu_lepton+s.mu_b-s.mu_a


@lru_cache(maxsize=16)
def nodes(order):
    return leggauss(order)


def angular_factor(Ee, cosine):
    beta=math.sqrt(max(0.,1-ME*ME/(Ee*Ee)))
    return GV**2*(1+beta*cosine)+GA**2*(3-beta*cosine)


def _energy_panels(state, Enu, cosine, tail):
    s=state.baryons
    E0=Enu-(s.offset_b-s.offset_a)
    central=max(ME,E0)
    p0=math.sqrt(max(0.,central*central-ME*ME))
    q0=math.sqrt(max(0.,Enu*Enu+p0*p0-2*Enu*p0*cosine))
    thermal=q0*math.sqrt(s.T/s.mass)
    # Recoil displacement and occupied Fermi sea both broaden energy support.
    broad=q0*math.sqrt(2*max(s.kinetic_mu_a,s.kinetic_mu_b,0.)/s.mass)
    recoil=q0*q0/(2*s.mass)
    width=max(thermal,broad,recoil,1e-12)
    upper=max(ME,E0,state.mu_lepton)+tail*s.T+12*width+recoil
    candidates=[ME,upper,E0,E0-recoil]
    candidates.extend(E0-recoil+k*width for k in (-16,-8,-4,-2,-1,1,2,4,8,16))
    candidates.extend(state.mu_lepton+k*s.T for k in (-12,-4,0,4,12))
    return sorted(set(x for x in candidates if ME<=x<=upper))


def log_coefficients(state: OpticalState, Enu, energy_order=24, angle_order=24, tail=40.):
    """Return (log j, log true chi), both coefficients in s^-1.

    Positive lepton-energy channels only: captures and their inverse neutrino
    absorption. Does not include neutron beta decay or its 3-body inverse.
    """
    if not math.isfinite(Enu) or Enu<=0:
        raise ValueError("Require finite Enu>0")
    if energy_order<4 or angle_order<4 or not math.isfinite(tail) or tail<=0:
        raise ValueError("Invalid quadrature controls")
    cs,cws=nodes(angle_order)
    xs,xws=nodes(energy_order)
    terms_j,terms_chi=[],[]
    s=state.baryons
    sr=s.reverse()
    for c,cw in zip(cs,cws):
        edges=_energy_panels(state,Enu,float(c),tail)
        for lo,hi in zip(edges[:-1],edges[1:]):
            Es=lo+.5*(xs+1)*(hi-lo)
            ws=.5*(hi-lo)*xws
            for Ee,ew in zip(Es,ws):
                p=math.sqrt(max(0.,Ee*Ee-ME*ME))
                q=math.sqrt(Enu*Enu+p*p-2*Enu*p*c)
                omega=Enu-Ee
                common=math.log(float(cw*ew*Ee*p*angular_factor(Ee,c)))
                vacancy=log_expit((Ee-state.mu_lepton)/s.T)
                occupation=log_expit((state.mu_lepton-Ee)/s.T)
                terms_chi.append(common+vacancy+log_response(s,q,omega))
                terms_j.append(common+occupation+log_response(sr,q,-omega))
    return float(math.log(K)+logsumexp(terms_j)),float(math.log(K)+logsumexp(terms_chi))


def coefficients(state, Enu, **kwargs):
    """j, chi [s^-1], true absorption chi/c [cm^-1], relaxation(j+chi)/c."""
    lj,lc=log_coefficients(state,Enu,**kwargs)
    j,chi=math.exp(lj),math.exp(lc)
    return {"j_s-1":j,"chi_s-1":chi,"true_absorption_cm-1":chi/C_CM_S,
            "relaxation_cm-1":(j+chi)/C_CM_S}
