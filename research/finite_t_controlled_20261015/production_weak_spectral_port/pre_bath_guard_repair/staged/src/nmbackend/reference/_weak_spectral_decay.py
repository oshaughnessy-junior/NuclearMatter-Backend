"""Conditional moving-neutron decay/inverse at fixed laboratory antinu energy.

Electron–proton pair phase, actual boosted FD products and no lab E0 cutoff.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from . import _weak_current as C
from ._weak_quadrature import log_fd
from ._weak_spectral_common import SpectralResult, coefficient, validate

REST_ENDPOINT = (C.MN-C.MP-C.ME)*(C.MN+C.MP+C.ME)/(2*C.MN)


def integrate(channel,bath,energy,controls,timer):
    energy = validate(channel,bath,energy,controls)
    if channel != 'NEUTRON_DECAY':
        raise ValueError('Named neutron decay required')
    eta0 = max(0.,math.log(energy/REST_ENDPOINT))
    pmin = C.MN*math.sinh(eta0); kmin = 2*C.MN*math.sinh(eta0/2)**2
    upper = max(kmin,0.,bath.mu_n-C.MN,bath.mu_p+bath.mu_e+energy-C.MN)+controls.tail*bath.T
    pmax = math.sqrt(upper*(2*C.MN+upper)); scale = math.sqrt(2*C.MN*bath.T)
    cuts = {pmin,pmax}|{v*scale for v in (.4,1,2,3.5,5,7) if pmin < v*scale < pmax}
    if energy < REST_ENDPOINT:
        switch = C.MN*math.sinh(math.log(REST_ENDPOINT/energy))
        if pmin < switch < pmax: cuts.add(switch)
    cuts = sorted(cuts)
    xp,wp = leggauss(controls.parent_order)
    xc,wc = leggauss(controls.cone_order); xe,we = leggauss(controls.electron_order)
    angle = (xc+1)*math.pi/4; angle_w = wc*math.pi/4
    phi = 2*math.pi*np.arange(controls.decay_phi_order)/controls.decay_phi_order
    jf=[];cb=[];calls=0; max_shell=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if not hi > lo:
            raise ArithmeticError('Collapsed parent panel, unresolved')
        for p,pw in zip(lo+(xp+1)*(hi-lo)/2,wp*(hi-lo)/2):
            timer.check()
            if not lo < p < hi or pw <= 0:
                raise ArithmeticError('Unrepresentable parent panel')
            en=math.hypot(C.MN,p); gamma=en/C.MN; velocity=p/en
            if velocity <= 0:
                raise ArithmeticError('Parent zero-velocity endpoint requires a separate limit')
            lower=max(-1.,(energy/(gamma*REST_ENDPOINT)-1)/velocity)
            if lower >= 1:
                raise ArithmeticError('Positive parent cone collapsed, unresolved')
            cs=lower+(1-lower)*np.sin(angle)**2
            cone_w=(1-lower)*2*np.sin(angle)*np.cos(angle)*angle_w/2
            den=gamma*(1+velocity*cs); ev=energy/den
            ds=2*C.MN*(REST_ENDPOINT-ev); sep=C.MN*C.MN-2*C.MN*ev
            if np.any(ds <= 0) or np.any(sep <= (C.MP+C.ME)**2-1e-9):
                raise ArithmeticError('Conditional pair support unresolved')
            rt=np.sqrt(sep); pr=np.sqrt(ds*(ds+4*C.MP*C.ME)/(4*sep)); es=np.hypot(pr,C.ME)
            gp=(C.MN-ev)/rt; vp=ev/(C.MN-ev)
            # cone × electron-CM cosine; rotation phi is only needed by bath FD.
            cc,et=np.meshgrid(cs,xe,indexing='ij'); ew=np.broadcast_to(we,(len(cs),len(xe)))
            evv=np.broadcast_to(ev[:,None],cc.shape); ppair=np.broadcast_to(pr[:,None],cc.shape)
            ee=gp[:,None]*(es[:,None]-vp[:,None]*ppair*et)
            px=ppair*np.sqrt(1-et*et); pz=gp[:,None]*(ppair*et-vp[:,None]*es[:,None])
            n=np.tile([C.MN,0,0,0],(cc.size,1))
            nu=np.column_stack([evv.ravel(),np.zeros(cc.size),np.zeros(cc.size),evv.ravel()])
            lepton=np.column_stack([ee.ravel(),px.ravel(),np.zeros(cc.size),pz.ravel()]); proton=n-lepton-nu
            amp=C.decay_amplitude(n,proton,lepton,nu).reshape(cc.shape)
            shell=float(np.max(abs(C.dot(proton,proton)/C.MP**2-1))); max_shell=max(max_shell,shell)
            if shell > 1e-10 or np.any(proton[:,0] < C.MP):
                raise ArithmeticError('Conditional proton shell unresolved')
            parent=p*p*pw/(math.pi**2*C.HC**3)
            measure=(cone_w[:,None]/den[:,None]*(ev*pr/rt)[:,None]*ew)*amp/(64*math.pi**3*C.MN*C.HBAR)*parent/gamma
            if np.any(measure <= 0) or not np.all(np.isfinite(measure)):
                raise ArithmeticError('Conditional positive phase unresolved')
            base=np.log(measure).ravel()-math.log(controls.decay_phi_order)
            cflat=cc.ravel(); sn=np.sqrt(1-cflat*cflat)
            # Parent direction relative to rest neutrino gives the same rest
            # rotation to each daughter before the lab boost.
            zzl=-sn[:,None]*lepton[:,1,None]*np.cos(phi)+cflat[:,None]*lepton[:,3,None]
            zzp=-sn[:,None]*proton[:,1,None]*np.cos(phi)+cflat[:,None]*proton[:,3,None]
            el=gamma*(lepton[:,0,None]+velocity*zzl)
            ep=gamma*(proton[:,0,None]+velocity*zzp)
            if np.max(abs(en-ep-el-energy))/max(en,energy) > 1e-10:
                raise ArithmeticError('Boosted conditional energy conservation failed')
            fn=float(log_fd((bath.mu_n-en)/bath.T)); hn=float(log_fd((en-bath.mu_n)/bath.T))
            fp=log_fd((bath.mu_p-ep)/bath.T); hp=log_fd((ep-bath.mu_p)/bath.T)
            fe=log_fd((bath.mu_e-el)/bath.T); he=log_fd((el-bath.mu_e)/bath.T)
            jf.append(float(logsumexp(base[:,None]+fn+hp+he)))
            cb.append(float(logsumexp(base[:,None]+hn+fp+fe)))
            calls+=cc.size*controls.decay_phi_order
    if not jf or not cb:
        raise ArithmeticError('Finite-T conditional spectrum unresolved, not exact zero')
    ld=2*math.log(energy)-math.log(2*math.pi**2*C.HC**3)
    timer.check()
    return SpectralResult(channel,bath,energy,coefficient(float(logsumexp(jf))-ld),coefficient(float(logsumexp(cb))-ld),
                          controls,('conditional lab E; ep pair; theta cone map',tuple(cuts),pmin,REST_ENDPOINT,calls,max_shell))
