"""Endpoint-smoothed quadrature for the frozen allowed capture model.

Identical physics/panels; Ee=me+x*x and exact Jacobian remove square-root
endpoint behavior. Selected-state numerical qualification only.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
from scipy.special import log_expit, logsumexp
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'allowed_opacity'))
from allowed_capture import OpticalState, ME, K, nodes, angular_factor, _energy_panels
from free_density_response import log_response

def log_coefficients(state: OpticalState, Enu, energy_order=48, angle_order=48, tail=40.):
    """Return (log j, log true chi), both coefficients in s^-1.

    Positive lepton-energy channels only: captures and their inverse neutrino
    absorption. Does not include neutron beta decay or its 3-body inverse.
    """
    if not math.isfinite(Enu) or Enu<=0:
        raise ValueError("Require finite Enu>0")
    if (not isinstance(energy_order,int) or not isinstance(angle_order,int)
        or isinstance(energy_order,bool) or isinstance(angle_order,bool)
        or energy_order<4 or angle_order<4 or not math.isfinite(tail) or tail<=0):
        raise ValueError("Invalid quadrature controls")
    cs,cws=nodes(angle_order)
    xs,xws=nodes(energy_order)
    terms_j,terms_chi=[],[]
    s=state.baryons
    sr=s.reverse()
    for c,cw in zip(cs,cws):
        edges=_energy_panels(state,Enu,float(c),tail)
        for lo,hi in zip(edges[:-1],edges[1:]):
            xlo=math.sqrt(max(0.,lo-ME))
            xhi=math.sqrt(max(0.,hi-ME))
            xx=xlo+.5*(xs+1)*(xhi-xlo)
            Es=ME+xx*xx
            ws=.5*(xhi-xlo)*xws*2*xx
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


