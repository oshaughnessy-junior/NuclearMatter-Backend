"""Leaf-local numerical guard for unrepresentable mapped energy panels.

Same capture_endpoint equations, panels and mapped quadrature. Discards only
panels whose sqrt(E-me) endpoints are identical in float64; exactpe uses x.
Frozen source is preserved; this extension requires independent review.
"""
from __future__ import annotations
import math
from scipy.special import log_expit,logsumexp
import mapped_capture as frozen
from allowed_capture import ME,K,nodes,angular_factor,_energy_panels
from free_density_response import log_response


def validate_controls(Enu,energy_order,angle_order,tail):
    """Reject invalid inputs before any numerical fallback is considered."""
    if not math.isfinite(Enu) or Enu<=0:
        raise ValueError('Require finite Enu>0')
    if (not isinstance(energy_order,int) or not isinstance(angle_order,int)
        or isinstance(energy_order,bool) or isinstance(angle_order,bool)
        or energy_order<4 or angle_order<4 or not math.isfinite(tail) or tail<=0):
        raise ValueError('Invalid quadrature controls')


def log_coefficients(state,Enu,energy_order=48,angle_order=48,tail=40.):
    validate_controls(Enu,energy_order,angle_order,tail)
    cs,cws=nodes(angle_order);xs,xws=nodes(energy_order)
    sj,sc=[],[];discarded=[];s=state.baryons;sr=s.reverse()
    for c,cw in zip(cs,cws):
        edges=_energy_panels(state,Enu,float(c),tail)
        for lo,hi in zip(edges[:-1],edges[1:]):
            xl=math.sqrt(max(0.,lo-ME));xh=math.sqrt(max(0.,hi-ME))
            if xh==xl:
                discarded.append(dict(cosine=float(c),lo=lo,hi=hi,width=hi-lo))
                continue
            xx=xl+.5*(xs+1)*(xh-xl)
            Es=ME+xx*xx;weights=.5*(xh-xl)*xws*2*xx
            for x,Ee,ew in zip(xx,Es,weights):
                # Algebraically exact for Ee=me+x², avoiding subtraction at me.
                pe=float(x)*math.sqrt(2*ME+float(x*x))
                q=math.sqrt(Enu*Enu+pe*pe-2*Enu*pe*c);omega=Enu-Ee
                common=sum(math.log(float(v)) for v in (cw,ew,Ee,pe,angular_factor(Ee,c)))
                sc.append(common+log_expit((Ee-state.mu_lepton)/s.T)+log_response(s,q,omega))
                sj.append(common+log_expit((state.mu_lepton-Ee)/s.T)+log_response(sr,q,-omega))
    return float(math.log(K)+logsumexp(sj)),float(math.log(K)+logsumexp(sc)),discarded
